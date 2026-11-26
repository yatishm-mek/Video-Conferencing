"""Speech-to-text: provider factory (local faster-whisper | Groq) + hallucination filter.

Pipecat's WhisperSTTService is a *segmented* STT: it only transcribes the audio
between VADUserStartedSpeakingFrame and VADUserStoppedSpeakingFrame, so Whisper
never sees silence. That satisfies "only send audio to Whisper when VAD confirms
speech" without extra code.
"""

from __future__ import annotations

import functools
import glob
import os
import re
import sys
import threading
import time
from collections import deque

import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.stt_service import STTService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language

from interview_bot.config import Settings

# One WhisperModel per process, shared by every interview session. Loading
# large-v3-turbo takes seconds; doing it per WebRTC connection would add that
# to every call's start.
_MODEL_CACHE: dict[tuple[str, str, str], object] = {}
_MODEL_LOCK = threading.Lock()


class CachedWhisperSTTService(WhisperSTTService):
    """WhisperSTTService that reuses a process-wide model and sets beam size.

    Pipecat's service does not expose beam_size, and faster-whisper defaults to
    beam_size=5, which roughly multiplies decode time. We bind it on the model's
    transcribe() so Pipecat's own run_stt (no_speech_prob filtering, metrics,
    finalized flags) stays untouched.
    """

    def __init__(self, *, beam_size: int = 1, max_compression_ratio: float = 2.4, min_avg_logprob: float = -1.0, **kwargs):
        self._beam_size = beam_size
        self._max_compression_ratio = max_compression_ratio
        self._min_avg_logprob = min_avg_logprob
        super().__init__(**kwargs)

    def _load(self):
        key = (str(self._settings.model), self._device, self._compute_type)
        with _MODEL_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                self._model = cached
                return
            super()._load()  # loads + validates language
            try:
                _warmup(self._model)
            except Exception as e:
                # Typical on Windows: an NVIDIA GPU is present but the CUDA 12
                # libraries (cublas64_12.dll, cuDNN 9) aren't installed. The
                # failure only shows at the first transcription, so we test now
                # instead of failing on every candidate answer.
                if self._device == "cpu":
                    raise
                logger.warning(
                    f"Whisper could not run on the GPU ({e}). Falling back to CPU int8. "
                    "To use the GPU: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*"
                )
                self._device, self._compute_type = "cpu", "int8"
                super()._load()
                _warmup(self._model)
            model = self._model
            # temperature=0.0 turns off faster-whisper's retry ladder (up to 6
            # re-decodes at rising temperatures when output looks unsure). On
            # "umm"/noise that ladder took 2-6 s per segment; one pass is
            # bounded. Timestamps aren't used, so skip predicting them.
            model.transcribe = _drop_looping_segments(  # type: ignore[method-assign]
                functools.partial(
                    model.transcribe,
                    beam_size=self._beam_size,
                    temperature=0.0,
                    without_timestamps=True,
                ),
                self._max_compression_ratio,
                self._min_avg_logprob,
            )
            _MODEL_CACHE[key] = model
            device = getattr(getattr(model, "model", None), "device", self._device)
            logger.info(
                f"Whisper ready on {str(device).upper()} ({self._compute_type}), "
                f"beam_size={self._beam_size}"
            )


# When speech-to-text threw sound away as noise (unsure, looping, "Thank you." hallucination).
# Several in a short time = background voices or noise in the room (live, Nov 18: people
# talking nearby; every reply was cancelled by their sound and the bot was silent 90 s).
_DROPS: deque = deque(maxlen=20)


def note_dropped() -> None:
    _DROPS.append(time.monotonic())


def noisy_room(window_secs: float = 60.0, count: int = 3) -> bool:
    now = time.monotonic()
    return sum(1 for t in _DROPS if now - t <= window_secs) >= count


def _drop_looping_segments(transcribe, max_ratio: float, min_logprob: float = -1.0):
    """Drop segments where Whisper got stuck repeating itself.

    On noise, music or background voices Whisper can loop ("BG, BG, BG, ..."
    for 8 s of audio). Looping text compresses very well, so faster-whisper's
    per-segment compression_ratio flags it (its own default limit is 2.4).
    Normally a retry at higher temperature would catch this; we turned retries
    off for speed, so we filter here instead.
    """

    def wrapped(*args, **kwargs):
        segments, info = transcribe(*args, **kwargs)

        def keep():
            for seg in segments:
                ratio = getattr(seg, "compression_ratio", 0.0)
                if ratio > max_ratio:
                    logger.info(f"Dropped looping STT segment (ratio {ratio:.1f}): {seg.text[:60]!r}")
                    note_dropped()
                    continue
                logprob = getattr(seg, "avg_logprob", 0.0)
                if logprob < min_logprob:
                    logger.info(f"Dropped unsure STT segment (avg_logprob {logprob:.2f}): {seg.text[:60]!r}")
                    note_dropped()
                    continue
                yield seg

        return keep(), info

    return wrapped


def _warmup(model) -> None:
    """One tiny transcription: surfaces missing GPU libraries at startup and
    takes the first-call slowness out of the first real answer."""
    segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)
    list(segments)


def _add_windows_cuda_dirs() -> None:
    """Let ctranslate2 find CUDA DLLs installed as pip wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12)."""
    if sys.platform != "win32":
        return
    import site

    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    for root in roots:
        for d in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                continue
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


_add_windows_cuda_dirs()


def build_stt(settings: Settings, vocab: list[str]) -> STTService:
    """Build the configured STT service. `vocab` biases recognition of tech terms."""
    language = Language(settings.STT_LANGUAGE)
    prompt = ", ".join(vocab) if vocab else None

    if settings.STT_PROVIDER == "moonshine":
        # Reuse the fine-tuned Moonshine Vosk-WS server as a one-shot client.
        from interview_bot.services.stt_moonshine import MoonshineSTTService

        logger.info(
            f"[stt] ENGINE = Moonshine (fine-tuned) via {settings.MOONSHINE_WS_URL} — "
            f"the candidate's speech is transcribed by that WS server. If it is not reachable, "
            f"nothing the candidate says (including 'yes') will register and the interview won't start."
        )
        return MoonshineSTTService(
            url=settings.MOONSHINE_WS_URL,
            language=settings.STT_LANGUAGE,
            sample_rate=16000,
        )

    if settings.STT_PROVIDER == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("STT_PROVIDER=groq but GROQ_API_KEY is empty")
        return GroqSTTService(
            api_key=settings.GROQ_API_KEY,
            settings=GroqSTTService.Settings(
                model=settings.GROQ_STT_MODEL,
                language=language,
                prompt=prompt,
                temperature=0.0,
            ),
        )

    logger.info(
        f"[stt] ENGINE = Whisper (faster-whisper {settings.WHISPER_MODEL}, in-process on "
        f"{settings.WHISPER_DEVICE}) — no external server needed."
    )
    return CachedWhisperSTTService(
        # How long the turn may wait for the last segment's transcript after
        # Smart Turn says "done". Too low and a slow transcript lands after the
        # turn closed, splitting one answer into two turns.
        ttfs_p99_latency=settings.STT_TTFS_P99_SECS,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        beam_size=settings.WHISPER_BEAM_SIZE,
        max_compression_ratio=settings.WHISPER_MAX_COMPRESSION_RATIO,
        min_avg_logprob=settings.WHISPER_MIN_AVG_LOGPROB,
        settings=WhisperSTTService.Settings(
            model=settings.WHISPER_MODEL,
            language=language,
            no_speech_prob=settings.WHISPER_NO_SPEECH_PROB,
            initial_prompt=prompt,
        ),
    )


def preload_stt(settings: Settings) -> None:
    """Load Whisper at server start so the first call doesn't wait for it."""
    if settings.STT_PROVIDER == "local":
        build_stt(settings, [])


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s']+")


def normalize(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def is_hallucination(text: str, phrases: list[str]) -> bool:
    """True if a transcript segment is a known Whisper hallucination.

    Short phrases ("you", "thank you") only match the *whole* segment, so a
    real answer containing them is kept. Long, YouTube-style phrases (3+ words)
    also match when contained in the segment.
    """
    norm = normalize(text)
    if not norm:
        return True
    for p in phrases:
        p_norm = normalize(p)
        if norm == p_norm:
            return True
        if len(p_norm.split()) >= 3 and p_norm in norm:
            return True
    return False


# People really do say these several times in a row ("No, no, no, no, I'm ready" was
# dropped live as a Whisper loop while the bot was saying goodbye).
SPOKEN_REPEATS = {"no", "yes", "wait", "stop", "ok", "okay", "sorry", "hello", "please", "hi", "yeah", "sir"}


def is_repetition_loop(text: str, min_repeats: int = 4, min_words: int = 6) -> bool:
    """True if the transcript is mostly one short phrase on repeat
    ("let's go, let's go, let's go, ..."). Catches loops from any provider."""
    words = normalize(text).split()
    if len(words) < 12 and all(w in SPOKEN_REPEATS for w in set(words) - {"i", "m", "am", "ready", "i'm"}):
        return False
    if len(words) < min_words:
        return False
    for n in (1, 2, 3):
        i = 0
        while i + n <= len(words):
            gram, reps = words[i : i + n], 1
            j = i + n
            while words[j : j + n] == gram:
                reps += 1
                j += n
            if reps >= min_repeats and reps * n >= len(words) / 2:
                if all(w in SPOKEN_REPEATS for w in gram) and len(words) < 16:
                    # Live: "No, no, no, no, no, I want to continue." over the goodbye was
                    # dropped as a loop, so the bot hung up. People do say "no" five times.
                    i += 1
                    continue
                return True
            i += 1
    return False


def is_prompt_echo(text: str, vocab: list[str]) -> bool:
    """True if Whisper just read back its own vocab prompt.

    On unclear or very short audio, Whisper can output the initial_prompt
    ("CTC, WER, Whisper, fine-tuning,") instead of what was said. We drop a
    segment when it is a run of 3+ words copied verbatim from the prompt. A
    candidate listing two terms, or using terms inside a sentence, is kept.
    """
    if not vocab:
        return False
    norm = normalize(text)
    if len(norm.split()) < 3:
        return False
    return f" {norm} " in f" {normalize(' '.join(vocab))} "


class HallucinationFilter(FrameProcessor):
    """Drops TranscriptionFrames that are Whisper hallucinations. Works for both providers."""

    def __init__(self, phrases: list[str], vocab: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._phrases = phrases
        self._vocab = vocab or []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            if is_hallucination(frame.text, self._phrases):
                logger.info(f"Dropped likely STT hallucination: {frame.text!r}")
                note_dropped()
                return
            if is_repetition_loop(frame.text):
                logger.info(f"Dropped repeating STT output: {frame.text[:60]!r}")
                return
            if is_prompt_echo(frame.text, self._vocab):
                logger.info(f"Dropped STT prompt echo: {frame.text!r}")
                return
        await self.push_frame(frame, direction)
