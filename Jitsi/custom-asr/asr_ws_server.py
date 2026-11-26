"""
Vosk-compatible WebSocket ASR server backed by the fine-tuned Moonshine model.

Jigasi's VoskTranscriptionService connects here (one connection per conference
participant) and speaks a tiny protocol:

    client -> server : {"config" : {"sample_rate" : N}}      (text, once)
    client -> server : <raw 16-bit signed LE mono PCM>       (binary, streamed)
    client -> server : {"eof" : 1}                           (text, at end)
    server -> client : {"partial" : "..."}                   (interim, optional)
    server -> client : {"text" : "..."}                      (final, per utterance)

Because Jigasi gives us a separate connection per speaker, each connection is a
single speaker -- so interviewer and candidate are transcribed independently and
Jitsi tags the captions with the right name. No diarization needed here.

Pipeline per connection:
    PCM -> float32 -> resample to 16 kHz -> Silero VAD endpointing ->
    Moonshine (contextual-biased) transcription -> guardrails -> {"text": ...}

Run it with the Realtime-ASR virtualenv's Python (which has the working
torch+cu128 / transformers / scipy / torchaudio / silero stack). The missing
`websockets` package is vendored under ./libs so the original venv is untouched.
See run-asr.ps1.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
from math import gcd
from pathlib import Path

import numpy as np

# Vendored deps (websockets) live next to this file so we don't modify the
# Realtime-ASR virtualenv.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "libs"))

import websockets  # noqa: E402  (from ./libs)
from scipy.signal import resample_poly  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import asr_engine as E  # noqa: E402
import guardrails as GR  # noqa: E402

# Diagnostic: when ASR_DEBUG_DUMP=1, every utterance's 16 kHz audio (exactly
# what the model sees) is saved under ./debug so it can be inspected / re-run
# through the standalone pipeline to separate audio-quality from server issues.
DEBUG_DUMP = os.environ.get("ASR_DEBUG_DUMP", "0") not in ("0", "false", "False")
DEBUG_DIR = Path(os.environ.get("ASR_DEBUG_DIR", str((Path(__file__).resolve().parent / "debug"))))

# --- config (override via env) ----------------------------------------------
HOST = os.environ.get("ASR_HOST", "0.0.0.0")
PORT = int(os.environ.get("ASR_PORT", "2700"))
TARGET_SR = E.SAMPLE_RATE  # 16 kHz for Moonshine + Silero

VAD_THRESHOLD = float(os.environ.get("ASR_VAD_THRESHOLD", "0.5"))
SILENCE_MS = int(os.environ.get("ASR_SILENCE_MS", "600"))      # pause ends a turn
MIN_SPEECH_MS = int(os.environ.get("ASR_MIN_SPEECH_MS", "250"))
SPEECH_PAD_MS = int(os.environ.get("ASR_SPEECH_PAD_MS", "200"))
MAX_UTTERANCE_MS = int(os.environ.get("ASR_MAX_UTTERANCE_MS", "15000"))
# Emit interim "partial" captions every N ms WHILE a person is still speaking,
# so text shows up live (like the old Vosk engine) instead of only after a
# pause. 0 disables partials (finals only). Lower = more responsive but more
# model calls (more GPU/CPU load).
PARTIAL_MS = int(os.environ.get("ASR_PARTIAL_MS", "500"))
USE_BIAS = os.environ.get("ASR_USE_BIAS", "1") not in ("0", "false", "False")

# --- hallucination suppression ----------------------------------------------
# Moonshine/Whisper-style models emit stock phrases ("Thank you", "so", "you")
# when fed near-silent audio or brief non-speech noise that slips past the VAD.
# Two guards:
#   1) Energy gate: segments whose RMS is below ASR_MIN_RMS are treated as
#      silence and never transcribed. This is the primary fix and does not touch
#      real speech (which is far louder). Keep it low so soft speech is kept.
#   2) Whole-utterance phrase filter: if the ENTIRE decoded text (normalised:
#      lowercased, punctuation/space-stripped) exactly equals one of these stock
#      phrases, it is dropped. Only whole-output matches are dropped, so the same
#      words inside a longer real sentence are always kept. Deliberately
#      conservative — it lists video-style hallucinations, not everyday words.
MIN_RMS = float(os.environ.get("ASR_MIN_RMS", "0.0045"))


def _normalize_for_match(text: str) -> str:
    """Lowercase and strip punctuation/extra space for whole-output matching."""
    t = re.sub(r"[^a-z0-9 ]+", "", text.lower())
    return re.sub(r"\s+", " ", t).strip()


_DEFAULT_HALLUCINATIONS = (
    "thanks for watching|thank you for watching|thank you so much for watching|"
    "please subscribe|like and subscribe|subscribe to my channel|"
    "see you next time|see you in the next video"
)
# Normalised set of whole-output phrases to drop (see note above).
HALLUCINATION_PHRASES = {
    _normalize_for_match(p)
    for p in os.environ.get("ASR_HALLUCINATIONS", _DEFAULT_HALLUCINATIONS).split("|")
    if _normalize_for_match(p)
}


def _is_hallucination(text: str) -> bool:
    return _normalize_for_match(text) in HALLUCINATION_PHRASES


def _looks_repetitive(text: str) -> bool:
    """True for degenerate repetition-loop output like
    'search is search is search is …' — a model failure, not real speech.
    Only triggers on longer outputs so short real phrases are never dropped."""
    words = _normalize_for_match(text).split()
    n = len(words)
    if n < 6:
        return False
    uniq = len(set(words))
    if uniq / n < 0.35:
        return True
    # a single word (or tiny cycle) dominating the whole utterance
    most = max(words.count(w) for w in set(words))
    return most / n > 0.5


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))

FRAME = 512                       # Silero frame size @ 16 kHz (== 32 ms)
FRAME_MS = 1000 * FRAME // TARGET_SR

# --- globals (model loaded once; GPU calls serialized) ----------------------
_processor = None
_model = None
_trie = None
_terms = None
_guard = None
_gpu_lock = threading.Lock()


def _init():
    global _processor, _model, _trie, _terms, _guard
    # On a CPU-only host, give PyTorch ALL cores for inference — fastest, which
    # matters most here. A brief event-loop stall no longer drops the jigasi
    # connection (ping_timeout is None). Override with ASR_TORCH_THREADS (e.g.
    # try "2" = physical cores vs "4" to see which is faster on your box).
    if not torch.cuda.is_available():
        try:
            ncpu = os.cpu_count() or 1
            nthreads = int(os.environ.get("ASR_TORCH_THREADS", "0")) or ncpu
            torch.set_num_threads(nthreads)
            print(f"[server] CPU inference: torch threads = {torch.get_num_threads()} (of {ncpu})")
        except Exception as _e:
            print(f"[server] could not set torch threads (ignored): {_e}")
    _processor, _model = E.load_model()
    _trie, _terms = E.make_bias_processor()
    print(f"[server] skills: {E.VOCAB_PATH.name} "
          f"({sum(1 for t in _terms if t.get('valid'))} biasing terms)")
    _guard = GR.from_file(HERE / "banned_words.txt", mode="mask")
    # Warm up (first generate is slow).
    print("[server] warming up model...")
    E.transcribe(_model, _processor,
                 np.zeros(TARGET_SR, dtype=np.float32), _trie, _terms, USE_BIAS)
    print(f"[server] ready. bias={'on' if USE_BIAS else 'off'} "
          f"vad_gate={VAD_THRESHOLD} end_pause={SILENCE_MS}ms")


def _transcribe_blocking(audio_16k: np.ndarray) -> str:
    """Run Moonshine (+biasing +guardrails) under the GPU lock. Returns text."""
    with _gpu_lock:
        text, dt = E.transcribe(_model, _processor, audio_16k,
                                _trie, _terms, USE_BIAS)
    # Latency visibility: how long inference took vs the audio length. RTF > 1
    # means slower than real time (the main cause of perceived caption lag on
    # CPU). Watch this in the server console to tune.
    try:
        dur = len(audio_16k) / TARGET_SR
        print(f"[asr] infer {dt * 1000:.0f} ms for {dur:.1f}s audio "
              f"(RTF {dt / max(dur, 1e-6):.2f})", flush=True)
    except Exception:
        pass
    if not text:
        return ""
    text, _blocked = _guard.filter(text)
    return text.strip()


def _resample_to_16k(x: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return x
    g = gcd(src_sr, TARGET_SR)
    return resample_poly(x, TARGET_SR // g, src_sr // g).astype(np.float32)


class Connection:
    """Per-participant streaming VAD + utterance assembly."""

    def __init__(self, ws, peer):
        self.ws = ws
        self.peer = peer
        self.sr = 16000
        # Silero VAD, one isolated instance per connection.
        from silero_vad import load_silero_vad
        self.vad = load_silero_vad()
        self.vad.reset_states()

        self.resid = np.zeros(0, dtype=np.float32)     # leftover < FRAME samples
        self.preroll = []                              # padding before onset
        self.pad_frames = max(1, SPEECH_PAD_MS // FRAME_MS)
        self.hangover_frames = max(1, SILENCE_MS // FRAME_MS)
        self.min_frames = max(1, MIN_SPEECH_MS // FRAME_MS)
        self.max_frames = max(1, MAX_UTTERANCE_MS // FRAME_MS)

        self.utt = []            # list of 512-sample frames in the current turn
        self.in_speech = False
        self.silence_run = 0
        self.partial_frames = (PARTIAL_MS // FRAME_MS) if PARTIAL_MS > 0 else 0
        self._last_partial_len = 0   # utt length at last partial emit

    def _vad_prob(self, frame: np.ndarray) -> float:
        with torch.inference_mode():
            return float(self.vad(torch.from_numpy(frame), TARGET_SR).item())

    async def feed_pcm(self, data: bytes):
        """Handle one binary PCM chunk; may emit a final result."""
        if not data:
            return
        pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        audio = _resample_to_16k(pcm, self.sr)
        if audio.size == 0:
            return
        buf = np.concatenate([self.resid, audio])
        n = (len(buf) // FRAME) * FRAME
        frames, self.resid = buf[:n], buf[n:]
        for i in range(0, n, FRAME):
            await self._process_frame(frames[i:i + FRAME])

    async def _process_frame(self, frame: np.ndarray):
        voiced = self._vad_prob(frame) >= VAD_THRESHOLD
        if not self.in_speech:
            self.preroll.append(frame)
            if len(self.preroll) > self.pad_frames:
                self.preroll.pop(0)
            if voiced:
                self.in_speech = True
                self.silence_run = 0
                self.utt = list(self.preroll)
                self.preroll = []
                self._last_partial_len = 0
        else:
            self.utt.append(frame)
            self.silence_run = 0 if voiced else self.silence_run + 1
            # Interim caption while still speaking (never blocks the final path).
            if (self.partial_frames
                    and (len(self.utt) - self._last_partial_len) >= self.partial_frames):
                self._last_partial_len = len(self.utt)
                await self._emit_partial()
            if self.silence_run >= self.hangover_frames or len(self.utt) >= self.max_frames:
                await self._flush()

    async def _emit_partial(self):
        """Transcribe the utterance-so-far and send an interim {"partial": ...}.

        Fully guarded: any failure here is logged and ignored so the final
        {"text": ...} path is never affected.
        """
        try:
            if len(self.utt) < self.min_frames:
                return
            audio = np.concatenate(self.utt).astype(np.float32)
            text = await asyncio.to_thread(_transcribe_blocking, audio)
            if text:
                await self.ws.send(json.dumps({"partial": text}))
        except Exception as e:
            print(f"[server] partial failed (ignored): {e}")

    async def _flush(self):
        self.in_speech = False
        frames, self.utt = self.utt, []
        self.silence_run = 0
        self._last_partial_len = 0
        self.vad.reset_states()
        if len(frames) < self.min_frames:
            return
        audio = np.concatenate(frames).astype(np.float32)
        # Energy gate: near-silent segments are not transcribed, so the model
        # can't hallucinate stock phrases ("Thank you", "so") on silence/noise.
        rms = _rms(audio)
        if rms < MIN_RMS:
            print(f"[{self.peer}] (skipped silent segment, rms={rms:.4f})")
            return
        if DEBUG_DUMP:
            try:
                DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                import time as _t
                fn = DEBUG_DIR / f"{int(_t.time()*1000)}_{self.peer}_sr{self.sr}.wav"
                sf.write(str(fn), audio, TARGET_SR)
            except Exception as _e:
                print(f"[server] dump failed: {_e}")
        text = await asyncio.to_thread(_transcribe_blocking, audio)
        if text and _is_hallucination(text):
            print(f"[{self.peer}] (dropped hallucination: {text!r})")
            return
        if text and _looks_repetitive(text):
            print(f"[{self.peer}] (dropped repetition loop: {text!r})")
            return
        if text:
            print(f"[{self.peer}] {text}")
            await self.ws.send(json.dumps({"text": text}))

    async def finish(self):
        """eof: flush any in-progress utterance as a final."""
        if self.in_speech and self.utt:
            await self._flush()
        await self.ws.send(json.dumps({"text": ""}))


async def handler(ws):
    peer = f"{getattr(ws, 'remote_address', ('?',))[0]}"
    conn = Connection(ws, peer)
    print(f"[server] + connection {peer}")
    try:
        async for message in ws:
            if isinstance(message, (bytes, bytearray)):
                await conn.feed_pcm(bytes(message))
            else:
                try:
                    obj = json.loads(message)
                except Exception:
                    continue
                if "config" in obj:
                    sr = obj["config"].get("sample_rate")
                    if sr:
                        conn.sr = int(sr)
                        print(f"[server] {peer} sample_rate={conn.sr}")
                elif "eof" in obj:
                    await conn.finish()
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"[server] - connection {peer}")


async def amain():
    _init()
    # ping_interval keeps the jigasi<->server websocket alive during speech
    # pauses (otherwise jigasi's Jetty client idle-timeout closes it and
    # utterances are lost until it reconnects). ping_timeout=None means we do
    # NOT close the connection if a pong is late: on a busy CPU the event loop
    # can be momentarily blocked by inference, and a 'keepalive ping timeout'
    # disconnect there would drop a live interview. The TCP layer still tears
    # down genuinely dead connections.
    async with websockets.serve(
        handler, HOST, PORT, max_size=None,
        ping_interval=20, ping_timeout=None, close_timeout=5,
    ):
        print(f"[server] listening on ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[server] stopped.")
