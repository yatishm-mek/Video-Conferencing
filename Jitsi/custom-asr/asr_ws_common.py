"""Shared Vosk-protocol WebSocket ASR framing (engine-agnostic).

This is the reusable core of `asr_ws_server.py` (the Moonshine server) with the
*engine* factored out, so a second server (e.g. faster-whisper) can reuse the
exact same streaming VAD + utterance assembly + hallucination guards and only
supply a `transcribe(audio_16k_float32) -> (text, seconds)` callable.

Protocol (what jigasi speaks — identical to asr_ws_server.py):
    client -> server : {"config": {"sample_rate": N}}   (text, once)
    client -> server : <raw 16-bit signed LE mono PCM>   (binary, streamed)
    client -> server : {"eof": 1}                        (text, at end)
    server -> client : {"partial": "..."}                (interim, optional)
    server -> client : {"text": "..."}                   (final, per utterance)

The Moonshine server (asr_ws_server.py) is intentionally left untouched; this
module is new and only the Whisper server imports it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from math import gcd
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "libs"))

import websockets  # noqa: E402  (vendored under ./libs)
from scipy.signal import resample_poly  # noqa: E402

TARGET_SR = 16000  # Silero VAD + both ASR models run at 16 kHz

# --- framing config (same env knobs as asr_ws_server.py) --------------------
VAD_THRESHOLD = float(os.environ.get("ASR_VAD_THRESHOLD", "0.5"))
SILENCE_MS = int(os.environ.get("ASR_SILENCE_MS", "600"))
MIN_SPEECH_MS = int(os.environ.get("ASR_MIN_SPEECH_MS", "250"))
SPEECH_PAD_MS = int(os.environ.get("ASR_SPEECH_PAD_MS", "200"))
MAX_UTTERANCE_MS = int(os.environ.get("ASR_MAX_UTTERANCE_MS", "15000"))
PARTIAL_MS = int(os.environ.get("ASR_PARTIAL_MS", "500"))
MIN_RMS = float(os.environ.get("ASR_MIN_RMS", "0.0045"))

FRAME = 512
FRAME_MS = 1000 * FRAME // TARGET_SR

_DEFAULT_HALLUCINATIONS = (
    "thanks for watching|thank you for watching|thank you so much for watching|"
    "please subscribe|like and subscribe|subscribe to my channel|"
    "see you next time|see you in the next video"
)


def _normalize_for_match(text: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", "", text.lower())
    return re.sub(r"\s+", " ", t).strip()


HALLUCINATION_PHRASES = {
    _normalize_for_match(p)
    for p in os.environ.get("ASR_HALLUCINATIONS", _DEFAULT_HALLUCINATIONS).split("|")
    if _normalize_for_match(p)
}


def is_hallucination(text: str) -> bool:
    return _normalize_for_match(text) in HALLUCINATION_PHRASES


def looks_repetitive(text: str) -> bool:
    words = _normalize_for_match(text).split()
    n = len(words)
    if n < 6:
        return False
    if len(set(words)) / n < 0.35:
        return True
    most = max(words.count(w) for w in set(words))
    return most / n > 0.5


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def resample_to_16k(x: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return x
    g = gcd(src_sr, TARGET_SR)
    return resample_poly(x, TARGET_SR // g, src_sr // g).astype(np.float32)


# transcribe(audio_16k_float32) -> (text, seconds). Runs under the caller's GPU
# lock inside a worker thread (asyncio.to_thread), so it may block.
TranscribeFn = Callable[[np.ndarray], "tuple[str, float]"]


class Connection:
    """Per-participant streaming VAD + utterance assembly (engine injected)."""

    def __init__(self, ws, peer: str, transcribe: TranscribeFn):
        self.ws = ws
        self.peer = peer
        self.sr = 16000
        self._transcribe = transcribe

        import torch  # noqa: PLC0415 (import here so non-torch callers can import this module)
        from silero_vad import load_silero_vad

        self._torch = torch
        self.vad = load_silero_vad()
        self.vad.reset_states()

        self.resid = np.zeros(0, dtype=np.float32)
        self.preroll: list[np.ndarray] = []
        self.pad_frames = max(1, SPEECH_PAD_MS // FRAME_MS)
        self.hangover_frames = max(1, SILENCE_MS // FRAME_MS)
        self.min_frames = max(1, MIN_SPEECH_MS // FRAME_MS)
        self.max_frames = max(1, MAX_UTTERANCE_MS // FRAME_MS)

        self.utt: list[np.ndarray] = []
        self.in_speech = False
        self.silence_run = 0
        self.partial_frames = (PARTIAL_MS // FRAME_MS) if PARTIAL_MS > 0 else 0
        self._last_partial_len = 0

    def _vad_prob(self, frame: np.ndarray) -> float:
        with self._torch.inference_mode():
            return float(self.vad(self._torch.from_numpy(frame), TARGET_SR).item())

    async def feed_pcm(self, data: bytes):
        if not data:
            return
        pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        audio = resample_to_16k(pcm, self.sr)
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
            if (self.partial_frames
                    and (len(self.utt) - self._last_partial_len) >= self.partial_frames):
                self._last_partial_len = len(self.utt)
                await self._emit_partial()
            if self.silence_run >= self.hangover_frames or len(self.utt) >= self.max_frames:
                await self._flush()

    async def _emit_partial(self):
        try:
            if len(self.utt) < self.min_frames:
                return
            audio = np.concatenate(self.utt).astype(np.float32)
            text, _ = await asyncio.to_thread(self._transcribe, audio)
            if text:
                await self.ws.send(json.dumps({"partial": text}))
        except Exception as e:  # noqa: BLE001
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
        if rms(audio) < MIN_RMS:
            return
        text, _ = await asyncio.to_thread(self._transcribe, audio)
        if text and is_hallucination(text):
            print(f"[{self.peer}] (dropped hallucination: {text!r})")
            return
        if text and looks_repetitive(text):
            print(f"[{self.peer}] (dropped repetition loop: {text!r})")
            return
        if text:
            print(f"[{self.peer}] {text}")
            await self.ws.send(json.dumps({"text": text}))

    async def finish(self):
        if self.in_speech and self.utt:
            await self._flush()
        await self.ws.send(json.dumps({"text": ""}))


def make_handler(transcribe: TranscribeFn):
    async def handler(ws):
        peer = f"{getattr(ws, 'remote_address', ('?',))[0]}"
        conn = Connection(ws, peer, transcribe)
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

    return handler


async def serve(host: str, port: int, transcribe: TranscribeFn):
    async with websockets.serve(
        make_handler(transcribe), host, port, max_size=None,
        ping_interval=20, ping_timeout=None, close_timeout=5,
    ):
        print(f"[server] listening on ws://{host}:{port}")
        await asyncio.Future()
