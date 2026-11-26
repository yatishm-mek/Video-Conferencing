"""Orchestrator configuration — all hosts, ports and launch commands in one place.

Everything is overridable by environment variable so the same code runs on the
dev laptop and on a server without edits. Defaults match the current single-box
setup (Jitsi at 192.168.7.174:8443, models native on the Windows host).
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = .../EZY-Hire (this file is integration/orchestrator/config.py)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---- Network -------------------------------------------------------------
# The browser-facing Jitsi URL. Interviewer (human) or bot both join here.
PUBLIC_URL = _env("EZY_PUBLIC_URL", "https://192.168.7.174:8443").rstrip("/")

# Where the orchestrator itself listens.
HOST = _env("EZY_ORCH_HOST", "127.0.0.1")
PORT = int(_env("EZY_ORCH_PORT", "8800"))

# ASR router control endpoint (Phase B). The orchestrator tells it which engine
# (moonshine | whisper) jigasi should transcribe with for the next interview.
# The router runs on the Jitsi box (next to jigasi), so default its control URL to
# that host automatically from PUBLIC_URL. Override with EZY_ASR_ROUTER_CONTROL.
from urllib.parse import urlparse as _urlparse  # noqa: E402

_jitsi_host = _urlparse(PUBLIC_URL).hostname or "127.0.0.1"
ASR_ROUTER_CONTROL = _env("EZY_ASR_ROUTER_CONTROL", f"http://{_jitsi_host}:2770").rstrip("/")

# The bot's own STT (when it listens with Moonshine) connects to the ASR ROUTER
# on the Jitsi box (:2700), NOT the Moonshine backend directly (:2710). The router
# is the exposed entry point (firewall opens :2700, not :2710), binds 0.0.0.0,
# defaults to Moonshine, and proxies the same Vosk protocol to it — so the bot gets
# Moonshine through it, just like jigasi does. Override with EZY_MOONSHINE_WS_URL.
MOONSHINE_WS_URL = _env("EZY_MOONSHINE_WS_URL", f"ws://{_jitsi_host}:2700")

# Qwen cloned-voice TTS microservice (Phase C) — used by the bot, not the orchestrator,
# but recorded here so the launch response can report what it will use.
QWEN_TTS_URL = _env("EZY_QWEN_TTS_URL", "http://127.0.0.1:8099").rstrip("/")

# ---- Bot runner (Phase E) -----------------------------------------------
# The interview_bot lives in its own venv. To start a bot-in-Jitsi session the
# orchestrator runs:  <BOT_PYTHON> -m interview_bot.jitsi_run --room <room> ...
# from the interview-bot package dir.
INTERVIEW_BOT_DIR = Path(
    _env(
        "EZY_INTERVIEW_BOT_DIR",
        str(REPO_ROOT / "interview_bot" / "interview-bot-phase1" / "interview-bot"),
    )
)
# Default to that package's own .venv python (Windows layout); override if needed.
BOT_PYTHON = _env(
    "EZY_BOT_PYTHON",
    str(INTERVIEW_BOT_DIR / ".venv" / "Scripts" / "python.exe"),
)
BOT_DISPLAY_NAME = _env("EZY_BOT_DISPLAY_NAME", "AI Interviewer")

# ---- Static launcher -----------------------------------------------------
LAUNCHER_DIR = REPO_ROOT / "launcher"

# Interview plan that drives the recruiter's "next questions" panel.
INTERVIEW_PLAN = Path(
    _env("EZY_INTERVIEW_PLAN", str(INTERVIEW_BOT_DIR / "data" / "sample_plan.json"))
)

# ---- Mappings ------------------------------------------------------------
# Launcher STT ids -> ASR router engine names. "arabic" is a future stub.
STT_ENGINE = {
    "moonshine": "moonshine",
    "whisper": "whisper",
}

# URL flag the Jitsi web app's custom-config.js reads to auto-start recording.
AUTORECORD_PARAM = "ezyrec"

# ---- Voice preview samples ----------------------------------------------
# Launcher voice id -> a small wav to play as a preview. Cloned voices use the
# real person's ~10 s reference clip (always present); Rithika also has a
# generated sample. "standard" (Kokoro) has no clip.
_VC = REPO_ROOT / "Voice cloning" / "voice-cloning"
VOICE_SAMPLES = {
    "rithika": _VC / "output" / "voice_samples" / "run2_epoch6_smooth.wav",
    "aditi": _VC / "data" / "voice_datasets" / "aditi" / "ref.wav",
    "sharanya": _VC / "data" / "voice_datasets" / "sharanya" / "ref.wav",
    "suresh": _VC / "data" / "voice_datasets" / "suresh" / "ref.wav",
}
# Fallbacks if a primary sample is missing.
VOICE_SAMPLES_FALLBACK = {
    "rithika": _VC / "data" / "voice_dataset" / "ref.wav",
}
