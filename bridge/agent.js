/*
 * EZY-Hire Jitsi bridge — headless participant.
 *
 * Joins the Jitsi room with lib-jitsi-meet (the real, version-matched client from
 * the deployment), captures the candidate's remote audio to Python over a WebSocket,
 * and publishes the bot's TTS audio back as a fake microphone. All WebRTC is owned
 * by the browser; Python only speaks PCM over the WS.
 *
 * URL params: ?room=&name=&ws=&domain=
 *   domain = the Jitsi PUBLIC_URL, e.g. https://192.168.7.174:8443
 *
 * NOTE: this is the spike-first component. Validate join/publish/capture with a
 * HEADED browser against the real server before trusting it headless. XMPP domain
 * assumptions (meet.jitsi / muc.meet.jitsi) match docker-jitsi-meet defaults; if the
 * deployment overrides XMPP_DOMAIN, the loaded config.js values below take over.
 */

const P = Object.fromEntries(new URLSearchParams(location.search));
const ROOM = (P.room || 'test').toLowerCase();
const NAME = P.name || 'AI Interviewer';
const WS_URL = P.ws;
const DOMAIN = (P.domain || location.origin).replace(/\/$/, '');
// The candidate's display name. We capture ONLY this participant's mic, so a
// recruiter (or anyone else) in the room can't garble the single audio stream
// we send to the STT. Empty => capture the first non-service human.
const CANDIDATE = (P.candidate || '').trim().toLowerCase();

function log(...a) { try { console.log('[bridge]', ...a); } catch (e) {} }

function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src; s.onload = res; s.onerror = () => rej(new Error('load failed: ' + src));
    document.head.appendChild(s);
  });
}

// ---- WebSocket to Python (reconnecting) -----------------------------------
let ws = null, wsReady = false;
function connectWS(onFlush) {
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { wsReady = true; log('ws open'); };
  ws.onclose = () => { wsReady = false; log('ws closed; retrying'); setTimeout(() => connectWS(onFlush), 1000); };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      try { const m = JSON.parse(ev.data); if (m.type === 'flush') onFlush && onFlush(); } catch (e) {}
    } else {
      onFlush && onFlush.playback && null; // binary = bot PCM, handled by caller
      if (window.__ezyPlayback) window.__ezyPlayback(ev.data);
    }
  };
}
function wsSendPCM(buf) { if (wsReady) try { ws.send(buf); } catch (e) {} }
function wsSendCtl(obj) { if (wsReady) try { ws.send(JSON.stringify(obj)); } catch (e) {} }

// ---- Audio graph (capture + fake-mic playback) ----------------------------
let audioCtx = null, playbackNode = null, fakeMicStream = null;
// Chromium quirk: a MediaStreamAudioSourceNode made from a REMOTE PeerConnection
// track outputs silence unless that stream is ALSO attached to a playing media
// element. We keep the pull elements alive here so the AudioWorklet actually
// receives the candidate's samples (otherwise [mic-in] stays at -90 dBFS).
const _pullElements = [];

async function setupAudio() {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
  try { await audioCtx.resume(); } catch (e) {}
  await audioCtx.audioWorklet.addModule('worklets/capture.js');
  await audioCtx.audioWorklet.addModule('worklets/playback.js');

  // Playback (bot -> fake mic): worklet -> MediaStreamDestination -> fake mic track.
  playbackNode = new AudioWorkletNode(audioCtx, 'playback-processor');
  const dest = audioCtx.createMediaStreamDestination();
  playbackNode.connect(dest);
  fakeMicStream = dest.stream;
  window.__ezyPlayback = (buf) => { try { playbackNode.port.postMessage(buf, [buf]); } catch (e) {} };

  // Override getUserMedia so lib-jitsi-meet's createLocalTracks picks up the fake mic.
  const realGUM = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (constraints) => {
    if (constraints && constraints.audio && !constraints.video) {
      return fakeMicStream;                       // our bot audio as the "microphone"
    }
    return realGUM(constraints);
  };
}

function captureRemoteTrack(mediaStreamTrack, pname) {
  const stream = new MediaStream([mediaStreamTrack]);
  // Pull the remote track through a muted <audio> element, or Web Audio gets
  // silence from it (Chromium bug). Muted so it is NOT heard by the bot/room.
  const puller = new Audio();
  puller.srcObject = stream;
  puller.muted = true;
  puller.autoplay = true;
  puller.play().catch(() => {});
  _pullElements.push(puller);                       // keep a ref so it isn't GC'd

  const src = audioCtx.createMediaStreamSource(stream);
  const cap = new AudioWorkletNode(audioCtx, 'capture-processor');
  cap.port.onmessage = (e) => wsSendPCM(e.data);  // Int16 48k block -> Python
  src.connect(cap);
  // Keep the node alive without audible output.
  const sink = audioCtx.createGain(); sink.gain.value = 0; cap.connect(sink).connect(audioCtx.destination);
  log('capturing audio from', pname || 'candidate');
  wsSendCtl({ type: 'remote-audio-ready' });
}

// ---- Jitsi join -----------------------------------------------------------
async function join() {
  const JitsiMeetJS = window.JitsiMeetJS;
  JitsiMeetJS.init({ disableAudioLevels: true });
  JitsiMeetJS.setLogLevel(JitsiMeetJS.logLevels.ERROR);

  // Prefer the deployment's own config.js values; fall back to docker defaults.
  const cfg = window.config || {};
  const hosts = cfg.hosts || { domain: 'meet.jitsi', muc: 'muc.meet.jitsi' };
  const host = DOMAIN.replace(/^https?:\/\//, '');
  const serviceUrl = (cfg.websocket || `wss://${host}/xmpp-websocket`) + `?room=${ROOM}`;

  const connection = new JitsiMeetJS.JitsiConnection(null, null, {
    hosts: hosts,
    serviceUrl: serviceUrl,
    clientNode: 'http://jitsi.org/jitsimeet',
  });

  let conference = null;

  connection.addEventListener(JitsiMeetJS.events.connection.CONNECTION_ESTABLISHED, async () => {
    log('xmpp connected; joining', ROOM);
    conference = connection.initJitsiConference(ROOM, { openBridgeChannel: true });

    // We capture exactly ONE participant's mic — the candidate's. The recruiter,
    // Jibri (Recorder) and jigasi (Transcriber) must not be mixed into the single
    // stream we send to the STT, or it turns to garble. Because a participant's
    // display name can arrive AFTER their track, we stash pending tracks and
    // re-check whenever a name shows up.
    const _pending = new Map();   // participantId -> MediaStreamTrack (not yet captured)
    let _capturedPid = null;

    function _nameOf(pid) {
      try { const p = conference.getParticipantById(pid); return (p && p.getDisplayName && p.getDisplayName()) || ''; }
      catch (e) { return ''; }
    }
    function _isService(nm) { return /^(Recorder|Transcriber)$/i.test(nm || ''); }
    function _matchesCandidate(nm) {
      if (!CANDIDATE) return true;                    // no target -> first non-service human
      const a = (nm || '').trim().toLowerCase();
      if (!a) return false;
      return a === CANDIDATE || a.includes(CANDIDATE) || CANDIDATE.includes(a);
    }
    function _tryCapture(pid) {
      if (_capturedPid || !_pending.has(pid)) return;
      const nm = _nameOf(pid);
      if (_isService(nm)) { log('skipping service track:', nm); _pending.delete(pid); return; }
      // If we know the candidate's name, wait for the matching participant.
      if (!_matchesCandidate(nm)) {
        log('not the candidate (' + (nm || '?') + '), waiting for', CANDIDATE); return;
      }
      _capturedPid = pid;
      const mst = _pending.get(pid); _pending.delete(pid);
      captureRemoteTrack(mst, nm);                    // the candidate's audio only
    }
    function _recheckAll() { for (const pid of Array.from(_pending.keys())) _tryCapture(pid); }

    conference.on(JitsiMeetJS.events.conference.TRACK_ADDED, (track) => {
      if (track.isLocal() || track.getType() !== 'audio') return;
      const pid = (track.getParticipantId && track.getParticipantId()) || Math.random().toString(36);
      _pending.set(pid, track.getTrack());
      _tryCapture(pid);
    });
    // Display names and join order settle after the fact: re-evaluate pending tracks.
    conference.on(JitsiMeetJS.events.conference.DISPLAY_NAME_CHANGED, () => _recheckAll());
    // A human (the candidate) joined the room -> tell Python so the bot can start.
    conference.on(JitsiMeetJS.events.conference.USER_JOINED, (id, user) => {
      let name = ''; try { name = (user && user.getDisplayName && user.getDisplayName()) || ''; } catch (e) {}
      log('participant joined:', name || id);
      wsSendCtl({ type: 'participant-joined', id: id, name: name });
      _recheckAll();   // a name may now resolve a pending track to the candidate
    });
    conference.on(JitsiMeetJS.events.conference.USER_LEFT, (id) => {
      log('participant left:', id);
      wsSendCtl({ type: 'participant-left', id: id, count: conference.getParticipantCount() });
    });
    conference.on(JitsiMeetJS.events.conference.CONFERENCE_JOINED, async () => {
      log('conference joined; participants already present:', conference.getParticipantCount());
      conference.setDisplayName(NAME);
      // Publish the bot's fake mic.
      const tracks = await JitsiMeetJS.createLocalTracks({ devices: ['audio'] });
      for (const t of tracks) await conference.addTrack(t);
      log('published bot audio track');
      // If a candidate is ALREADY in the room when the bot joins, start now.
      if (conference.getParticipantCount() > 0) {
        try { conference.getParticipants().forEach(p => wsSendCtl({ type: 'participant-joined', id: p.getId(), name: p.getDisplayName() || '' })); } catch (e) {}
      }
    });
    conference.on(JitsiMeetJS.events.conference.CONFERENCE_LEFT, () => {
      log('conference left'); wsSendCtl({ type: 'conference-left' });
    });

    conference.join();
  });

  connection.addEventListener(JitsiMeetJS.events.connection.CONNECTION_FAILED, (e) => log('connection failed', e));
  connection.addEventListener(JitsiMeetJS.events.connection.CONNECTION_DISCONNECTED, () => log('disconnected'));

  connection.connect();
}

// ---- boot -----------------------------------------------------------------
(async function main() {
  try {
    log('loading Jitsi client from', DOMAIN);
    try { await loadScript(`${DOMAIN}/config.js`); } catch (e) { log('config.js not loaded (using defaults):', e.message); }
    await loadScript(`${DOMAIN}/libs/lib-jitsi-meet.min.js`);
    await setupAudio();
    connectWS(() => { if (playbackNode) playbackNode.port.postMessage({ cmd: 'flush' }); });
    await join();
  } catch (e) {
    log('fatal', e && e.message ? e.message : e);
  }
})();
