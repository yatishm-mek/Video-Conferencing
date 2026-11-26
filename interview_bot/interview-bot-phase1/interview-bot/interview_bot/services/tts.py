"""Text-to-speech: Kokoro-82M (kokoro-onnx), loaded once per process.

Pipecat's KokoroTTSService downloads (first run, ~350 MB) and loads the ONNX
model inside __init__, on the event loop. Done per WebRTC call, that freezes
the whole server for seconds and can drop the browser connection. We swap the
module's Kokoro class for a cached factory and preload it at server start.
"""

from __future__ import annotations

import asyncio
import re
import threading

import numpy as np
import pipecat.services.kokoro.tts as kokoro_mod
from loguru import logger
from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, KokoroTTSService

from interview_bot.config import Settings

_RealKokoro = kokoro_mod.Kokoro
_CACHE: dict[tuple[str, str], object] = {}
_LOCK = threading.Lock()


def _cached_kokoro(model_path: str, voices_path: str):
    key = (model_path, voices_path)
    with _LOCK:
        if key not in _CACHE:
            logger.info("Loading Kokoro TTS model...")
            _CACHE[key] = _RealKokoro(model_path, voices_path)
            logger.info("Kokoro TTS model loaded")
        return _CACHE[key]


kokoro_mod.Kokoro = _cached_kokoro  # KokoroTTSService looks the name up at call time


_SENT = re.compile(r"(?<=[.!?])\s+")


def _key(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(" ".join(text.split())) if s]


# Audio for lines known in advance, shared by every session: (voice, speed, line) -> (int16 bytes, rate).
_AUDIO: dict[tuple[str, float, str], tuple[bytes, int]] = {}


def _params(tts):
    rg = getattr(kokoro_mod, "require_given", lambda v, _n: v)
    ag = getattr(kokoro_mod, "assert_given", lambda v: v)
    return rg(tts._settings.voice, "Kokoro TTS voice"), rg(tts._settings.language, "Kokoro TTS language"), ag(tts._settings.speed)


def preload_lines(settings: Settings, texts: list[str], cache_dir=None) -> None:
    """Voice the planned questions and fixed lines once, at server start (off the event
    loop, before "Ready"). Doing it per call, during the greeting, took 45-56 s and starved
    the live voice: the bot said "Hi, Ganesh." and then nothing."""
    import time

    import hashlib
    from pathlib import Path

    from interview_bot.config import PROJECT_ROOT

    tts = build_tts(settings)
    voice, lang, speed = _params(tts)
    # Saved to disk (data/tts_cache): the next server start loads them in well under a
    # second instead of voicing 30+ lines again on the CPU (live: 50 s before "Ready").
    disk = Path(cache_dir) if cache_dir else PROJECT_ROOT / "data" / "tts_cache"
    disk.mkdir(parents=True, exist_ok=True)
    t0, n, loaded = time.monotonic(), 0, 0
    for t in texts:
        for s in sentences(t):
            key = (voice, speed, _key(s))
            if key in _AUDIO:
                continue
            f = disk / (hashlib.sha1(f"{voice}|{lang}|{speed}|{_key(s)}".encode()).hexdigest()[:20] + ".npz")
            if f.exists():
                try:
                    z = np.load(f)
                    _AUDIO[key] = (z["pcm"].tobytes(), int(z["rate"]))
                    loaded += 1
                    continue
                except Exception:  # noqa: BLE001 - a broken file is voiced again
                    pass
            samples, rate = tts._kokoro.create(s, voice=voice, lang=lang, speed=speed)
            pcm = (np.asarray(samples) * 32767).astype(np.int16)
            _AUDIO[key] = (pcm.tobytes(), rate)
            try:
                np.savez(f, pcm=pcm, rate=np.int64(rate))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[tts] could not save {f.name}: {e}")
            n += 1
    logger.info(f"[tts] {n} lines voiced, {loaded} loaded from data/tts_cache ({time.monotonic() - t0:.1f}s)")


class CachedKokoroTTSService(KokoroTTSService):
    """Kokoro that plays ready-made audio for lines voiced at server start (the planned
    questions, "Sure.", "Alright, let's begin." ...). Live: "Sure." played, then a gap of
    about a second while the question was voiced - "it makes my heart drop, did the bot
    fail?". Everything else is voiced as before."""

    async def run_tts(self, text: str, context_id: str):
        voice, _lang, speed = _params(self)
        hit = _AUDIO.get((voice, speed, _key(text)))
        if hit is None:
            async for f in super().run_tts(text, context_id):
                yield f
            return
        from pipecat.frames.frames import TTSAudioRawFrame

        await self.start_tts_usage_metrics(text)
        audio, rate = hit
        data = await self._resampler.resample(audio, rate, self.sample_rate)
        await self.stop_ttfb_metrics()
        step = int(self.sample_rate * 0.2) * 2  # 200 ms pieces, so it can still be interrupted
        for i in range(0, len(data), step):
            yield TTSAudioRawFrame(audio=data[i:i + step], sample_rate=self.sample_rate, num_channels=1,
                                   context_id=context_id)


def build_tts(settings: Settings):
    if settings.TTS_PROVIDER == "qwen":
        # Cloned voice via the Qwen3-TTS microservice (separate venv/process).
        from interview_bot.services.tts_qwen import QwenCloneTTSService

        return QwenCloneTTSService(
            url=settings.QWEN_TTS_URL, voice=settings.QWEN_VOICE, style=settings.QWEN_STYLE
        )
    return CachedKokoroTTSService(
        settings=KokoroTTSService.Settings(voice=settings.KOKORO_VOICE, speed=settings.KOKORO_SPEED)
    )


def preload_tts(settings: Settings) -> None:
    """Download (first run) and load Kokoro at server start, off the event loop."""
    if settings.TTS_PROVIDER == "qwen":
        # Cloned voices live in the separate TTS microservice; nothing to preload here.
        logger.info("TTS_PROVIDER=qwen: skipping Kokoro preload (using the cloned-voice microservice)")
        return
    KOKORO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    kokoro_mod._ensure_model_files(
        KOKORO_CACHE_DIR / "kokoro-v1.0.onnx", KOKORO_CACHE_DIR / "voices-v1.0.bin"
    )
    tts = build_tts(settings)  # also validates KOKORO_VOICE
    # The first synthesis is several times slower (ONNX warm-up): pay it now,
    # not on the candidate's greeting.
    tts._kokoro.create("Hello.", voice=settings.KOKORO_VOICE, lang="en-us")
    logger.info("Kokoro warmed up")
