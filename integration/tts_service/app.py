r"""Cloned-voice TTS microservice (Qwen3-TTS + LoRA).

A thin, synchronous HTTP wrapper around the voice-cloning `VoiceBank`, kept as its
own process because it needs the heavy `qwen3tts` venv + GPU and a ~4 GB base model
that must stay out of the interview bot's venv. The bot's Pipecat TTS adapter
(services/tts_qwen.py) POSTs text here and streams back the audio.

    POST /tts   {voice, text, style?}  ->  raw PCM16LE mono, header X-Sample-Rate: 24000
    GET  /voices                       ->  the available cloned voices
    GET  /health

Run from the qwen3tts venv:
    $env:EZY_TTS_PORT = "8099"
    "%USERPROFILE%\.venvs\qwen3tts\Scripts\python.exe" integration\tts_service\app.py

Serializes generation with one lock (single GPU), same as voice_studio.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    import torch  # same venv as VoiceBank; used to free GPU memory between calls
except Exception:  # noqa: BLE001
    torch = None

# Make the voice-cloning package importable and the base model offline-loadable.
REPO_ROOT = Path(__file__).resolve().parents[2]
VC_ROOT = Path(os.environ.get("EZY_VOICE_CLONING_DIR", str(REPO_ROOT / "Voice cloning" / "voice-cloning")))
sys.path.insert(0, str(VC_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from voice_finetune.voice_bank import VoiceBank  # noqa: E402

PORT = int(os.environ.get("EZY_TTS_PORT", "8099"))
HOST = os.environ.get("EZY_TTS_HOST", "127.0.0.1")
VOICES_JSON = VC_ROOT / "voice_studio" / "voices.json"

# Speaking styles (mirror voice_studio/server.py STYLES).
STYLES = {
    "steady": {"temperature": 0.6, "top_k": 30, "subtalker_temperature": 0.7},
    "balanced": {"temperature": 0.75, "top_k": 40, "subtalker_temperature": 0.8},
    "expressive": {"temperature": 0.9, "top_k": 50, "subtalker_temperature": 0.9},
}


def _load_voice_map() -> dict[str, dict]:
    """voice id -> {run_dir, epoch} from voices.json, with a hardcoded fallback."""
    try:
        data = json.loads(VOICES_JSON.read_text(encoding="utf-8"))
        out = {}
        for v in data.get("voices", []):
            out[v["id"]] = {"run_dir": VC_ROOT / v["run"], "epoch": int(v.get("epoch", 6))}
        if out:
            return out
    except Exception as e:  # noqa: BLE001
        print(f"[tts] could not read {VOICES_JSON} ({e}); using defaults")
    return {
        "rithika": {"run_dir": VC_ROOT / "output/voice_finetune/run2", "epoch": 6},
        "aditi": {"run_dir": VC_ROOT / "output/voice_finetune/aditi", "epoch": 6},
        "sharanya": {"run_dir": VC_ROOT / "output/voice_finetune/sharanya", "epoch": 6},
        "suresh": {"run_dir": VC_ROOT / "output/voice_finetune/suresh", "epoch": 6},
    }


VOICE_MAP = _load_voice_map()

# The raw base model has NO fixed speaker (its config ships spk_id = {}), so asking it
# for "the base voice" free-generates a DIFFERENT timbre on every call — that's the
# "many different voices" problem. Only the fine-tuned LoRA voices have a trained, stable
# speaker. So "standard"/"base" is pinned to ONE fixed fine-tuned voice for a single,
# consistent bot voice. Override which one with EZY_TTS_STANDARD_VOICE=<id>.
STANDARD_VOICE = os.environ.get("EZY_TTS_STANDARD_VOICE", "").strip()
if STANDARD_VOICE not in VOICE_MAP:
    STANDARD_VOICE = next(
        (v for v in ("aditi", "sharanya", "rithika", "suresh") if v in VOICE_MAP),
        next(iter(VOICE_MAP), ""),
    )

app = FastAPI(title="EZY-Hire Cloned-Voice TTS")
_bank: VoiceBank | None = None
_lock = threading.Lock()   # one generation at a time (single GPU)


@app.on_event("startup")
def _startup():
    global _bank
    print("[tts] loading Qwen3-TTS base model (once)...")
    _bank = VoiceBank(device=os.environ.get("EZY_TTS_DEVICE", "cuda:0"))
    print(f"[tts] ready on http://{HOST}:{PORT}; voices: {list(VOICE_MAP)}")
    print(f"[tts] 'standard'/'base' -> fixed voice '{STANDARD_VOICE or '(none: raw base)'}' "
          f"(set EZY_TTS_STANDARD_VOICE to change)")
    _log(f"[tts] started; {_gpu_mem()} (shared with Ollama — watch this grow across calls)")


class TTSRequest(BaseModel):
    voice: str
    text: str
    style: str = "balanced"
    max_new_tokens: int | None = None


LOG_PATH = Path(__file__).resolve().parent / "tts_service.log"


def _log(msg: str) -> None:
    """Print and append to a file, so a crash reason survives even if the console scrolls."""
    print(msg, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:  # noqa: BLE001
        pass


def _gpu_mem() -> str:
    if torch is None or not torch.cuda.is_available():
        return "cpu"
    try:
        free, total = torch.cuda.mem_get_info()
        return f"{(total - free) / 1024**3:.1f}/{total / 1024**3:.1f} GB used on GPU"
    except Exception:  # noqa: BLE001
        return "gpu mem unknown"


def _free_gpu() -> None:
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _synth(voice: str, text: str, style: str, max_new_tokens: int | None) -> tuple[bytes, int]:
    sampling = STYLES.get(style, STYLES["balanced"])
    with _lock:
        try:
            if voice in ("base", "standard"):
                # Unified voice: use one fixed fine-tuned voice, NOT the speaker-less base
                # model (which would give a different voice each utterance). See STANDARD_VOICE.
                spec = VOICE_MAP.get(STANDARD_VOICE)
                if spec:
                    _bank.use(spec["run_dir"], epoch=spec["epoch"])
                else:
                    _bank.use_base()  # no fine-tuned voices available: fall back to raw base
            else:
                spec = VOICE_MAP[voice]
                _bank.use(spec["run_dir"], epoch=spec["epoch"])
            wav, sr = _bank.speak(text, max_new_tokens=max_new_tokens, **sampling)
        except Exception as e:  # noqa: BLE001
            # Log the REAL reason (CUDA OOM, generation error, ...) with GPU usage, so the
            # crash the bot sees as an empty "synthesis failed ()" is actually diagnosable.
            _log(f"[tts] SYNTH FAILED for {voice!r} ({len(text)} chars): "
                 f"{type(e).__name__}: {e} | {_gpu_mem()}")
            _log(traceback.format_exc())
            _free_gpu()
            raise
        finally:
            # Release per-call activation memory so it doesn't creep up and OOM over a long
            # session — the GPU is shared with Ollama, so headroom is tight.
            _free_gpu()
    pcm = (np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return pcm, int(sr)


@app.post("/tts")
async def tts(req: TTSRequest):
    if req.voice not in VOICE_MAP and req.voice not in ("base", "standard"):
        return JSONResponse(status_code=404, content={"error": f"unknown voice '{req.voice}'", "choices": ["base"] + list(VOICE_MAP)})
    text = (req.text or "").strip()
    if not text:
        return Response(content=b"", media_type="application/octet-stream", headers={"X-Sample-Rate": "24000"})
    import anyio

    try:
        pcm, sr = await anyio.to_thread.run_sync(_synth, req.voice, text, req.style, req.max_new_tokens)
    except Exception as e:  # noqa: BLE001
        # Return a real error body instead of letting the exception reset the connection,
        # so the bot logs the reason (not an empty "synthesis failed ()").
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})
    return Response(content=pcm, media_type="application/octet-stream", headers={"X-Sample-Rate": str(sr)})


@app.get("/voices")
async def voices():
    return {"voices": list(VOICE_MAP)}


@app.get("/health")
async def health():
    return {"ok": _bank is not None, "voices": list(VOICE_MAP)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
