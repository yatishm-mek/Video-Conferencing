# Jitsi Meet with per-speaker transcription (interviewer vs. candidate)

A local, Docker-based Jitsi Meet deployment configured so that each participant's
audio is captured and transcribed **separately** — e.g. one stream for the
interviewer and one for the candidate.

> **Transcription engine:** this now uses the fine-tuned **Moonshine** model from
> `D:\Realtime-ASR` (contextual biasing + guardrails), served natively on the GPU.
> See [`custom-asr/README.md`](../custom-asr/README.md). Start `custom-asr\run-asr.ps1`
> before beginning transcription in a call. The original local **Vosk** engine
> (below) remains available as a fallback.

## How the two separate streams work

Jitsi's video bridge (**JVB**) is an SFU, not a mixer — it never combines audio.
Every participant is already an independent RTP/WebRTC stream. This setup adds a
**Jigasi transcriber** bot that joins each meeting, receives each participant's
audio as its own stream, and feeds each to a local **Vosk** speech-to-text
engine. The result:

- **Live captions** tagged per speaker (shown via the CC / subtitles button).
- A **per-meeting transcript** (`.txt`) with each line attributed to a speaker.
- A **per-speaker audio recording** (one `.wav` per participant) — the literal
  "two separate audio streams".

```
Interviewer ─┐
Candidate   ─┤──►  JVB (keeps each audio stream separate)
             │          │
          Jicofo ──►  Jigasi transcriber (joins the call)
                         │  each participant = its own audio stream
                         ▼
                   Vosk STT  (ws://vosk:2700, offline, free, no API key)
                         │
                         ▼
          per-speaker captions + transcript.txt + per-speaker .wav
```

Everything runs offline. No cloud account or API key is required.

## Layout

| Path | What it is |
|------|------------|
| `docker-jitsi-meet/` | Official `docker-jitsi-meet`, pinned to tag `stable-11248` |
| `docker-jitsi-meet/.env` | Configuration (passwords generated, transcription enabled) |
| `docker-jitsi-meet/vosk.yml` | Adds the offline Vosk STT container |
| `docker-jitsi-meet/jitsi.ps1` | Helper: `up` / `down` / `logs` / `ps` |
| `docker-jitsi-meet/.jitsi-meet-cfg/` | Generated runtime config + outputs |
| `docker-jitsi-meet/.jitsi-meet-cfg/storage/transcripts/` | Saved transcripts and per-speaker `.wav` files |

## Prerequisites

- **Docker Desktop** running (WSL2 backend). Give it **at least 6–8 GB RAM** —
  the Jitsi stack plus the Vosk English model is memory-hungry.
- A browser (Chrome/Edge/Firefox). Mic access requires HTTPS, which this setup
  serves with a self-signed certificate on `https://localhost:8443`.

## Start it

From a PowerShell prompt in `docker-jitsi-meet/`:

```powershell
.\jitsi.ps1 up
```

The first run downloads several GB of images (Jitsi components + the Vosk
English model) and can take a while. Watch progress with:

```powershell
.\jitsi.ps1 logs
```

Then open **https://localhost:8443** and accept the self-signed certificate
warning.

## Run an interview

> **Start the Moonshine ASR server first** (`custom-asr/run-asr.ps1`) so the
> transcriber has an engine to talk to — otherwise captions stay empty.

1. Open `https://localhost:8443` and create a room, e.g. `interview1`.
   Enter your display name (**required** — e.g. **Interviewer**).
2. In a second tab/window (or another device — see note below), open the same
   room `https://localhost:8443/interview1` and enter the name **Candidate**.
3. **Transcription starts automatically** when you join — there is no CC button
   to press (it has been removed). Lines are tagged with each speaker's real name
   (the display name entered in step 1/2) and are saved to the transcript / sent
   to the in-call chat.
4. To **record the video**, open **⋮ (More actions) → Start recording →
   Record**. This is optional and independent of transcription (see
   [Video recording (Jibri)](#video-recording-jibri)).
5. After the meeting ends, find outputs under:
   `docker-jitsi-meet/.jitsi-meet-cfg/storage/transcripts/`
   - `*.txt` — transcript with **per-speaker lines and per-line timestamps**
     (e.g. `<9:12:18 AM> abc: Can we continue the interview from here`)
   - `*.wav` — one audio file per participant

   and video recordings under:
   `docker-jitsi-meet/.jitsi-meet-cfg/storage/jibri/Video Recordings/`

### Saving / finding transcripts

Transcripts are saved **automatically** for every meeting (no action needed) and
they **persist on the host** — each meeting gets its own folder under
`.jitsi-meet-cfg/storage/transcripts/`, holding the timestamped `.txt` and the
per-speaker `.wav` files. Those folders are named with UUIDs, so to collect them
into a clean, readable, dated set run:

```powershell
.\jitsi.ps1 transcripts
```

This copies every meeting's transcript into
`.jitsi-meet-cfg/storage/transcripts/Interview Transcripts/` named
`<room>_<YYYY-MM-DD_HH-MM-SS>.txt` (e.g. `interview_2026-11-16_09-07-52.txt`).
Originals are left untouched and the command is safe to re-run after each
interview — it only adds new ones.

### Auto-transcription & speaker names

- **Auto-start on join, no button:** `docker-jitsi-meet/.jitsi-meet-cfg/web/custom-config.js`
  (appended to Jitsi's `config.js`) requests transcription as soon as you join —
  this version of Jitsi has no built-in "start captions on join" flag. The
  manual **CC / Subtitles** button is removed from the toolbar via `TOOLBAR_BUTTONS`
  in `.env` (the standard list minus `closedcaptions`), so transcription is fully
  automatic. To revert: add `closedcaptions` back to `TOOLBAR_BUTTONS`, and/or
  empty `custom-config.js`, then recreate the `web` container.
- **Real names instead of "Fellow Jitser":**
  - `ENABLE_REQUIRE_DISPLAY_NAME=1` forces every **human** to enter a name before
    joining, so their transcript lines are attributed correctly.
  - The other "Fellow Jitser" you may have seen is the **Jibri recorder's**
    headless browser, which joins nameless whenever you record. `custom-config.js`
    now detects the recorder (`config.iAmRecorder`) and names it **"Recorder"**,
    so it no longer appears as "Fellow Jitser" in tiles or the transcript.
- **Latency is hardware-bound here — read this.** The Moonshine ASR server uses a
  CUDA GPU if present, else **CPU**. This machine is an **Intel Core i3-3220
  (2012, 2 cores / 4 threads, AVX only — no AVX2, no GPU)**. Moonshine is a
  transformer; on this CPU each utterance takes real compute time, so there is a
  **latency floor no setting can remove**. The server now logs each inference as
  `[asr] infer <ms> for <n>s audio (RTF <x>)` — an **RTF above ~1 means slower
  than real time**, which is the lag you see. The genuinely effective fixes:
  1. **Run the ASR server on a machine with a GPU** (what the original setup did)
     — this is by far the biggest win. Point `JIGASI_TRANSCRIBER_VOSK_URL` in
     `.env` at that host.
  2. **int8 quantization** (`ASR_QUANTIZE=1`) — 2-3x faster, but **only on a CPU
     with AVX2** (this one lacks it and would crash; left off).
  3. **A smaller Moonshine model** (base/tiny) — much faster, but needs
     re-fine-tuning, so not a drop-in here.
  On CPU you must also optimise for *completeness*, not live latency — otherwise
  the transcriber falls behind and **drops audio**. Tunables in
  `custom-asr/run-asr.ps1`:
  - `ASR_PARTIAL_MS` — interim "while you speak" captions. Each partial re-runs
    the model on the growing utterance, so on CPU they multiply load and cause
    dropped audio. **Set to `0` (disabled) on a CPU box.** Only raise it above 0
    if you move the ASR server to a machine with a CUDA GPU.
  - `ASR_SILENCE_MS` — end-of-utterance pause before the final line. `500` is a
    safe balance on CPU; raise toward `600` if lines get chopped mid-sentence,
    lower only on a GPU box for snappier finals.
  - `ASR_MAX_UTTERANCE_MS` — `8000` so long monologues flush progressively instead
    of waiting up to 15 s.
- **Missing / incomplete transcription?** Two common causes:
  1. **CPU overload from partials** — set `ASR_PARTIAL_MS=0` (see above).
  2. **WebSocket idle-timeout drops.** If the jigasi→ASR connection goes idle
     (long pause, or the CPU is busy), jigasi's client closes it and every
     utterance until it reconnects is lost. The server now sends keepalive pings
     (`ping_interval` in `asr_ws_server.py`) to prevent this. Check for it with:
     `docker compose logs transcriber | grep "Idle Timeout"` — it should be empty.
  - Make sure the Moonshine server is actually running and reachable at the
    `JIGASI_TRANSCRIBER_VOSK_URL` host in `.env`
    (`curl -i http://<host>:2700` should return `426 Upgrade Required`).
- **Reducing latency (CPU).** With partials off, perceived latency ≈
  `ASR_SILENCE_MS` + model inference time. Knobs in `custom-asr/run-asr.ps1`:
  - `ASR_SILENCE_MS` — captions appear ~this long after you stop talking. `400`
    is responsive; `300` is snappier but can chop sentences.
  - `ASR_MAX_UTTERANCE_MS` — `6000`; shorter audio per call = faster CPU inference.
  - `ASR_TORCH_THREADS` — PyTorch CPU threads; `0`/unset uses all cores
    (the server prints the count on startup).
- **Model hallucinating ("Thank you", "so") when nobody speaks?** Whisper/Moonshine
  models emit stock phrases on silence or brief noise. The ASR server suppresses
  these two ways (in `custom-asr/asr_ws_server.py`):
  - **Energy gate** `ASR_MIN_RMS` (default `0.0045`): near-silent segments are
    never transcribed, so silence can't produce text. Raise it slightly if
    hallucinations persist on background noise; lower it if soft speech is
    getting dropped. The server logs `(skipped silent segment, rms=…)`.
  - **Whole-utterance phrase filter** `ASR_HALLUCINATIONS` (pipe-separated): drops
    an utterance whose *entire* text is a known hallucination (e.g.
    `thanks for watching`). Only exact whole-output matches are dropped, so the
    same words inside a real sentence are always kept. Everyday words like
    "thank you"/"so"/"okay" are deliberately **not** listed — the energy gate
    handles those when they're hallucinated on silence, while real ones (spoken
    with voice energy) pass through.
- **Repetition loops ("search is search is search is …")?** Moonshine can get
  stuck repeating a phrase on unclear audio. This both prints garbage and wastes
  CPU time (latency), and on a slow box can stall the server enough to drop the
  jigasi WebSocket (`keepalive ping timeout` in the ASR log). It's handled two
  ways: generation uses `repetition_penalty`/`no_repeat_ngram_size`
  (`ASR_REPETITION_PENALTY`, `ASR_NO_REPEAT_NGRAM`) and a lower token cap
  (`ASR_MAX_NEW_TOKENS=64`) to stop loops early, and a post-filter drops any
  output that is mostly one repeated word/phrase.
- **ASR connection drops / `keepalive ping timeout`?** The server no longer
  closes a connection when a pong is late (`ping_timeout=None`) — on a busy CPU
  the event loop can stall briefly, and dropping a live interview over that is
  worse than a late pong. It also leaves one CPU core free for the event loop
  (`ASR_TORCH_THREADS`, default = cores − 1) so it stays responsive during
  inference. The `NNPACK ... Unsupported hardware` lines in the log are harmless.

> **After changing `.env`, `custom-config.js`, or the ASR server you must:**
> 1. recreate the stack — `docker compose down` then `docker compose up -d`;
> 2. restart the Moonshine server (`run-asr.ps1`);
> 3. **hard-refresh the browser (Ctrl+Shift+R)** — browsers cache `config.js`
>    aggressively, so a normal reload can keep serving the old settings.

## Video recording (Jibri)

Full-meeting **video recording** is provided by **Jibri**, Jitsi's built-in
recorder. Jibri joins the call headlessly (a hidden Chromium instance), records
the composite video + audio to an `.mp4`, and on completion a finalize script
moves the file into a dedicated **`Video Recordings`** folder.

```
Interviewer ─┐
Candidate   ─┤──►  JVB ──► (meeting)
             │                 ▲
          Jicofo ──────────────┘ hands the room to an idle Jibri
             │
             ▼
   Jibri (headless Chromium + ffmpeg)  ──►  <id>.mp4
             │  finalize.sh runs when recording stops
             ▼
   .jitsi-meet-cfg/storage/jibri/Video Recordings/<room>_<timestamp>_<id>.mp4
```

It is wired into the stack via `docker-jitsi-meet/jibri.yml` (already added to
`jitsi.ps1`), enabled with `ENABLE_RECORDING=1` in `.env`, and uses PulseAudio's
virtual sink — so **no host kernel module** (`snd_aloop`) is required.

**How to record**

1. Start the stack (`.\jitsi.ps1 up`) and give Jibri ~30–60 s after the other
   containers to register with Jicofo.
2. Join a room, open the **⋮ (More actions)** menu in the toolbar and click
   **Start recording** → **Record** (local/file recording).
3. Click **Stop recording** (or leave the call). Within a few seconds the
   finished file appears here, inside `docker-jitsi-meet/`:

   ```
   .jitsi-meet-cfg/storage/jibri/Video Recordings/
       <room>_<YYYYmmdd-HHMMSS>_<id>.mp4
       <room>_<YYYYmmdd-HHMMSS>_metadata.json
   ```

   The raw per-session folders remain under
   `.jitsi-meet-cfg/storage/jibri/recordings/` until you clear them.

**Notes**

- **Quality:** set `JIBRI_RECORDING_RESOLUTION` (default `1280x720`) and
  `JIBRI_RECORDING_FRAMERATE` (default `30`) in `.env`.
- **Change the output folder name** or layout by editing the `DEST` path in
  `docker-jitsi-meet/.jitsi-meet-cfg/jibri/finalize.sh`.
- **Jibri needs memory and a 2 GB `/dev/shm`** (configured via `shm_size` in
  `jibri.yml`). Give Docker Desktop **8 GB+** when recording alongside the Vosk
  transcription model.
- **No "Start recording" button?** Confirm `ENABLE_RECORDING=1` in `.env` and
  recreate the stack (`down` then `up`) — prosody must re-run to register the
  `recorder` XMPP user.
- **Recording never starts / "unavailable"?** Check `.\jitsi.ps1 logs jibri`;
  Jibri must finish registering in the `jibribrewery` MUC before it can accept a
  request. Confirm the Java app is actually up:
  `docker compose exec jibri curl -s http://localhost:2222/jibri/api/v1.0/health`
  should return `IDLE / HEALTHY`. If Jicofo logs say *"Unable to find an available
  Jibri"* even though the container is "Up", the cause is usually the stock
  startup script's **blocking Chrome pre-warm** (`google-chrome ... about:blank`),
  which hangs on hosts without a working GPU (`VK_ERROR_INCOMPATIBLE_DRIVER` /
  "Page load timed out") and prevents Jibri's Java process from ever launching.
  This repo ships a patched startup script
  (`.jitsi-meet-cfg/jibri/jibri-start.sh`, mounted over
  `/etc/s6-overlay/scripts/jibri` via `jibri.yml`) that runs the pre-warm in the
  background under a `timeout`, so Jibri always starts and registers. If you reset
  `.jitsi-meet-cfg/`, keep that script (like `finalize.sh`).

## Review UI — watch recordings with a synced transcript

The [`review-ui/`](review-ui/) folder is a small, dependency-free web app to
browse finished interviews. For each meeting it shows:

- the **meeting name** (the Jitsi room),
- **who joined** (participant names, bots excluded),
- a **video card** that plays the recording, and
- the **transcript** below it — **click any line and the video jumps to that
  moment**, and the line under the playhead is highlighted as the video plays.

**Start it** (needs Python 3; nothing to install):

```powershell
cd review-ui
.\run-ui.ps1
```

On startup it prints two URLs — one for **this machine** (`http://localhost:8080`)
and one for **the network** (`http://<this-machine-ip>:8080`, e.g.
`http://192.168.7.174:8080`). Open either; use the IP one to reach it from the
machine's IP or another device on the LAN. (Plain `python3 server.py` works too.)

It reads the live storage folders on every refresh, so **a new interview appears
as soon as you record it** — just hit **↻ Refresh**. No build step, no database.

How it works:

- Transcripts come from `.jitsi-meet-cfg/storage/transcripts/` (saved
  automatically by jigasi, with per-line timestamps).
- Recordings come from `.jitsi-meet-cfg/storage/jibri/Video Recordings/`.
- `finalize.sh` writes a small `<video>.info.json` **sidecar** next to each
  recording with the video's exact start time (UTC epoch) and duration; that's
  what lets the UI line up transcript timestamps with the video position. (Older
  recordings without a sidecar still work — the server reads the mp4's duration
  and file time as a fallback, just slightly less precise.)
- **Timezones are handled automatically.** jigasi writes transcript times in the
  container's timezone (usually UTC), which may differ from the host. The server
  anchors each transcript to the UTC timestamp in its folder name, so line→video
  offsets are correct no matter how the container and host clocks differ. The
  card's date/time is shown exactly as the transcript labels read.
- The server (`review-ui/server.py`, Python stdlib only) pairs each transcript
  with its recording by room + overlapping time, computes each line's offset into
  the video, and serves the video with HTTP range requests so seeking works.

Config via env vars: `REVIEW_UI_PORT` (default `8080`), `REVIEW_UI_HOST`
(default `0.0.0.0` — reachable at the machine's IP and from the LAN; set
`127.0.0.1` to restrict it to this machine only), and `JITSI_STORAGE` to point at
a different `.jitsi-meet-cfg/storage` path.

## Stop it

```powershell
.\jitsi.ps1 down
```

## Notes & tuning

- **Testing from another device on your LAN:** set `JVB_ADVERTISE_IPS` in `.env`
  to your machine's LAN IP (e.g. `192.168.1.50`) and use that IP in the URL. For
  same-machine two-tab testing, the current `127.0.0.1` is correct.
- **Language:** the Vosk image is English (`alphacep/kaldi-en`). For another
  language, change the image in `vosk.yml` to a different Vosk model server and
  restart.
- **Turn off per-speaker WAV recording:** set `JIGASI_TRANSCRIBER_RECORD_AUDIO=false`
  in `.env` if you only want transcripts.
- **Config changes** in `.env` require a recreate: `.\jitsi.ps1 down` then
  `.\jitsi.ps1 up`.
- **Reset everything:** `.\jitsi.ps1 down` then delete the `.jitsi-meet-cfg/`
  directory (it is regenerated on next `up`). **Note:** this also deletes the
  Jibri finalize script (`.jitsi-meet-cfg/jibri/finalize.sh`) and any saved
  recordings under `.jitsi-meet-cfg/storage/jibri/` — back those up first. The
  finalize script and `Video Recordings` folder are recreated automatically only
  if you keep them; otherwise re-add `finalize.sh` **and `jibri-start.sh`** (see
  this repo) before recording again — without `jibri-start.sh`, Jibri may fail to
  start on GPU-less hosts and recording will show "unavailable".
