"""Vosk-protocol WebSocket ASR server backed by faster-whisper.

The Whisper counterpart to asr_ws_server.py (Moonshine). It speaks the exact same
Vosk protocol on a different port (default 2701), so jigasi can transcribe with
Whisper instead of Moonshine simply by being pointed at this server (directly, or
via asr_router.py). All the streaming VAD / endpointing / hallucination guards are
shared from asr_ws_common.py; only the transcription engine differs.

Run with a venv that has faster-whisper + torch + silero-vad + scipy + soundfile.

    $env:WHISPER_ASR_PORT = "2701"
    $env:WHISPER_MODEL     = "deepdml/faster-whisper-large-v3-turbo-ct2"
    python whisper_ws_server.py
"""

from __future__ import annotations

import glob
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

import asr_ws_common as C
import guardrails as GR

HERE = Path(__file__).resolve().parent


def _add_windows_cuda_dirs() -> None:
    """Let ctranslate2 (faster-whisper) find CUDA DLLs installed as pip wheels
    (nvidia-cublas-cu12, nvidia-cudnn-cu12) on Windows GPU."""
    if sys.platform != "win32":
        return
    import site

    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001
        pass
    for root in roots:
        for d in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                continue
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


_add_windows_cuda_dirs()

HOST = os.environ.get("WHISPER_ASR_HOST", "0.0.0.0")
PORT = int(os.environ.get("WHISPER_ASR_PORT", "2701"))
MODEL = os.environ.get("WHISPER_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
DEVICE = os.environ.get("WHISPER_DEVICE", "auto")          # auto | cpu | cuda
COMPUTE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")   # int8 on CPU; int8_float16 on CUDA
BEAM = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))
LANG = os.environ.get("WHISPER_LANGUAGE", "en")
# Same tech-term biasing prompt the interview bot uses (helps domain words).
PROMPT = os.environ.get("WHISPER_PROMPT", "").strip() or None

_model = None
_guard = None
_gpu_lock = threading.Lock()


def _init():
    global _model, _guard
    from faster_whisper import WhisperModel

    device, compute = DEVICE, COMPUTE
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"[server] loading faster-whisper {MODEL} on {device} ({compute})...")
    _model = WhisperModel(MODEL, device=device, compute_type=compute)
    _guard = GR.from_file(HERE / "banned_words.txt", mode="mask")
    # Warm up (first decode is slow).
    list(_model.transcribe(np.zeros(C.TARGET_SR, dtype=np.float32), language=LANG, beam_size=1)[0])
    print(f"[server] ready. model={MODEL} beam={BEAM} lang={LANG}")


def _transcribe(audio_16k: np.ndarray) -> tuple[str, float]:
    """faster-whisper transcription under the GPU lock. Returns (text, seconds)."""
    t0 = time.monotonic()
    with _gpu_lock:
        segments, _info = _model.transcribe(
            audio_16k,
            language=LANG,
            beam_size=BEAM,
            temperature=0.0,            # no retry ladder: bounded latency
            without_timestamps=True,
            initial_prompt=PROMPT,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    dt = time.monotonic() - t0
    try:
        dur = len(audio_16k) / C.TARGET_SR
        print(f"[asr] infer {dt*1000:.0f} ms for {dur:.1f}s audio (RTF {dt/max(dur,1e-6):.2f})", flush=True)
    except Exception:
        pass
    if not text:
        return "", dt
    text, _blocked = _guard.filter(text)
    return text.strip(), dt


async def amain():
    _init()
    await C.serve(HOST, PORT, _transcribe)


if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[server] stopped.")
