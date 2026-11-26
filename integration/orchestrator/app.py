"""EZY-Hire orchestrator.

One entry point that the launcher page talks to. It turns a launcher payload
(interview type + STT/LLM/TTS choices) into concrete actions:

  Human interview:
    - set the ASR router's engine (so jigasi transcribes with the chosen model)
    - return the Jitsi room URL for the interviewer to open
    - recording stays manual (the Jitsi toolbar button)

  Bot interview:
    - set the ASR router's engine
    - start the bot as a headless Jitsi participant (interview_bot.jitsi_run)
    - return the Jitsi room URL with the auto-record flag set
    - recording is automatic (custom-config.js starts Jibri on join)

It also serves launcher/index.html at "/", so the whole thing is one origin.

Run:  python -m uvicorn integration.orchestrator.app:app --host 127.0.0.1 --port 8800
  (or: python integration/orchestrator/app.py)

Nothing here needs a GPU; it only coordinates the other services.
"""

from __future__ import annotations

import json
import logging
import subprocess
import os
import sys
import time
from collections import defaultdict
from typing import Literal, Optional
from urllib.parse import quote
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from integration.orchestrator import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orchestrator")

app = FastAPI(title="EZY-Hire Orchestrator")

# Allow the launcher to call the API even when opened from file:// or another
# origin (the browser sends Origin: null for file://). No credentials, so "*" is fine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track bot processes we started, keyed by room, so a second launch can't double-join.
_bot_procs: dict[str, subprocess.Popen] = {}

# In-memory live transcript per room: list of {speaker, text, ts}. The recruiter
# page polls GET /api/transcript/{room}; a tap on the ASR layer (or the Jitsi
# captions, client-side) can POST lines in.
_transcripts: dict[str, list[dict]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class LaunchRequest(BaseModel):
    interview_type: Literal["human", "bot"]
    candidate: str = ""
    jitsi_room: str = Field(..., min_length=1)
    stt: str = "moonshine"            # moonshine | whisper  (arabic = future)
    llm: Optional[str] = None         # own | openai   (bot only)
    tts: Optional[str] = None         # kokoro | standard | rithika | aditi | sharanya | suresh (bot only)
    style: str = "balanced"           # steady | balanced | expressive (cloned voices only)
    recruiter: str = "Recruiter"      # recruiter display name (userId in the URL)
    profile_id: Optional[str] = None  # shared candidate profile id
    recruiter_id: Optional[str] = None
    plan: str = ""                    # candidate plan slug (data/candidates/<slug>)


class LaunchResponse(BaseModel):
    ok: bool
    recruiter_url: str
    candidate_url: str
    interview_type: str
    engine: str
    recording: str
    bot_started: bool = False
    notes: list[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interview_urls(request: Request, req: "LaunchRequest", auto_record: bool) -> tuple[str, str]:
    """Build the recruiter and candidate URLs to OUR interview page.

    The page is served by this orchestrator (so /interview.html is a real page, not a
    Jitsi room) and embeds Jitsi from PUBLIC_URL. Host and Jitsi host can differ.

      http://<this-host>/interview.html?type=recruiter&userId=&profileId=&recruiterId=&room=&plan=&jitsi=
      http://<this-host>/interview.html?type=candidate&userId=&profileId=&room=&plan=&jitsi=
    """
    base = str(request.base_url).rstrip("/")
    room = req.jitsi_room
    profile_id = req.profile_id or uuid4().hex
    recruiter_id = req.recruiter_id or uuid4().hex
    bot = req.interview_type == "bot"
    common = (
        f"&room={quote(room)}&itype={req.interview_type}&stt={quote(req.stt)}"
        f"&plan={quote(req.plan)}&jitsi={quote(cfg.PUBLIC_URL)}"
    )
    rec = f"&{cfg.AUTORECORD_PARAM}=1" if auto_record else ""
    recruiter = (
        f"{base}/interview.html?type=recruiter&userId={quote(req.recruiter or 'Recruiter')}"
        f"&profileId={profile_id}&recruiterId={recruiter_id}{common}{rec}"
    )
    candidate = (
        f"{base}/interview.html?type=candidate&userId={quote((req.candidate or 'Candidate').replace(' ', '_'))}"
        f"&profileId={profile_id}{common}{rec}"
    )
    if bot:
        recruiter += f"&llm={quote(req.llm or 'own')}&tts={quote(req.tts or 'standard')}"
    return recruiter, candidate


async def _set_asr_engine(engine: str) -> None:
    """Tell the ASR router which backend jigasi should transcribe with.

    Fire-and-forget + short timeout so a missing/slow router never delays the
    launch (the engine only matters once captions start, seconds later anyway).
    """
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            r = await client.post(f"{cfg.ASR_ROUTER_CONTROL}/set_engine", json={"engine": engine})
            r.raise_for_status()
        log.info("ASR router engine set to %s", engine)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not set ASR engine (%s): %s", engine, e)


def _start_bot(req: LaunchRequest, notes: list[str]) -> bool:
    """Launch the bot as a headless Jitsi participant for this room."""
    if req.jitsi_room in _bot_procs and _bot_procs[req.jitsi_room].poll() is None:
        notes.append("A bot is already running for this room; not starting another.")
        return True

    cmd = [
        cfg.BOT_PYTHON,
        "-m",
        "interview_bot.jitsi_run",
        "--room",
        req.jitsi_room,
        "--name",
        cfg.BOT_DISPLAY_NAME,
        "--stt",
        req.stt,
        "--tts",
        req.tts or "standard",
        "--style",
        req.style or "balanced",
        "--llm",
        req.llm or "own",
    ]
    if req.plan:
        cmd += ["--plan", f"data/candidates/{req.plan}/plan.json"]
    if req.candidate:
        cmd += ["--candidate", req.candidate]

    # Point the bot's Moonshine STT at the SERVER's Moonshine WS (the Jitsi box),
    # not the bot host's localhost. The bot reads MOONSHINE_WS_URL from its env.
    env = {**os.environ, "MOONSHINE_WS_URL": cfg.MOONSHINE_WS_URL}
    try:
        proc = subprocess.Popen(cmd, cwd=str(cfg.INTERVIEW_BOT_DIR), env=env)
        _bot_procs[req.jitsi_room] = proc
        log.info(
            "Started bot for room %s (pid %s): %s  [MOONSHINE_WS_URL=%s]",
            req.jitsi_room, proc.pid, " ".join(cmd), cfg.MOONSHINE_WS_URL,
        )
        return True
    except FileNotFoundError:
        notes.append(
            f"Bot python not found at {cfg.BOT_PYTHON}. Set EZY_BOT_PYTHON to the interview-bot venv python."
        )
        return False
    except Exception as e:  # noqa: BLE001
        notes.append(f"Failed to start bot: {e}")
        return False


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.post("/api/launch", response_model=LaunchResponse)
async def launch(req: LaunchRequest, request: Request):
    notes: list[str] = []
    engine = cfg.STT_ENGINE.get(req.stt)
    if engine is None:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"STT '{req.stt}' is not available yet (arabic is planned)."},
        )

    # Fire-and-forget so Start returns immediately (don't block the meeting launch).
    import asyncio

    asyncio.create_task(_set_asr_engine(engine))

    is_bot = req.interview_type == "bot"
    bot_started = False
    if is_bot:
        bot_started = _start_bot(req, notes)

    recruiter_url, candidate_url = _interview_urls(request, req, auto_record=is_bot)
    return LaunchResponse(
        ok=True,
        recruiter_url=recruiter_url,
        candidate_url=candidate_url,
        interview_type=req.interview_type,
        engine=engine,
        recording="automatic" if is_bot else "manual",
        bot_started=bot_started,
        notes=notes,
    )


@app.get("/api/health")
async def health():
    return {"ok": True, "public_url": cfg.PUBLIC_URL}


@app.get("/api/config")
async def client_config():
    """What the embedded-interview page needs: the Jitsi host to embed."""
    return {"public_url": cfg.PUBLIC_URL, "autorecord_param": cfg.AUTORECORD_PARAM}


@app.get("/video-interview/{room}")
async def video_interview(room: str):
    """Serve the embedded-Jitsi interview page (recruiter or candidate view).

    The view is chosen client-side from ?type=; the room is in the path.
    """
    return FileResponse(cfg.LAUNCHER_DIR / "interview.html")


@app.get("/api/questions/{room}")
async def questions(room: str):
    """The interview plan's questions, for the recruiter's 'next questions' panel."""
    try:
        plan = json.loads(cfg.INTERVIEW_PLAN.read_text(encoding="utf-8"))
        qs = [
            {"id": q.get("id"), "topic": q.get("topic", ""), "q": q.get("q", ""),
             "expected": q.get("expected", [])}
            for q in plan.get("questions", [])
        ]
        return {"ok": True, "candidate": plan.get("candidate", ""), "role": plan.get("role", ""), "questions": qs}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=200, content={"ok": False, "error": str(e), "questions": []})


class TranscriptLine(BaseModel):
    speaker: str = ""
    text: str


@app.get("/api/transcript/{room}")
async def get_transcript(room: str, since: int = 0):
    """Live transcript lines for a room (poll with ?since=<index>)."""
    lines = _transcripts.get(room, [])
    return {"ok": True, "lines": lines[since:], "next": len(lines)}


@app.post("/api/transcript/{room}")
async def post_transcript(room: str, line: TranscriptLine):
    """Append a transcript line (ingest from an ASR tap or the Jitsi captions)."""
    _transcripts[room].append({"speaker": line.speaker, "text": line.text, "ts": time.time()})
    return {"ok": True, "count": len(_transcripts[room])}


class EvaluateRequest(BaseModel):
    question: str
    context: str = ""
    expected: list[str] = []
    answer: str
    level: str = ""
    vocab: list[str] = []
    earlier: list[list[str]] = []


@app.post("/api/evaluate")
async def api_evaluate(req: EvaluateRequest):
    """Score one answer with the same LLM judge the bot uses (for human interviews)."""
    from integration.orchestrator import evaluate as _ev

    if not req.answer.strip():
        return {"ok": False, "error": "empty answer"}
    try:
        out = await _ev.evaluate(req.model_dump())
        return {"ok": True, **out}
    except Exception as e:  # noqa: BLE001
        log.warning("evaluate failed: %s", e)
        return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})


@app.get("/api/voice-sample/{voice}")
async def voice_sample(voice: str):
    """A short wav preview for a cloned voice (the person's reference clip)."""
    path = cfg.VOICE_SAMPLES.get(voice)
    if path is None or not path.is_file():
        path = cfg.VOICE_SAMPLES_FALLBACK.get(voice)
    if path is None or not path.is_file():
        return JSONResponse(status_code=404, content={"ok": False, "error": f"no sample for '{voice}'"})
    return FileResponse(path, media_type="audio/wav")


def _pcm_to_wav(pcm: bytes, rate: int, channels: int = 1) -> bytes:
    """Wrap raw 16-bit PCM in a minimal WAV container so the browser can play it."""
    import struct

    byte_rate = rate * channels * 2
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate, byte_rate, channels * 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


# Fixed phrase each voice says in a generated preview.
PREVIEW_TEXT = "Hi, this is how I will sound in the interview."


@app.get("/api/voice-preview/{voice}")
async def voice_preview(voice: str, style: str = "balanced"):
    """Generate a short clip in the ACTUAL voice + style via the TTS microservice,
    so previews reflect what the bot will really sound like (and differ per voice).
    Falls back to the static sample if the TTS service isn't running."""
    qvoice = "base" if voice in ("standard", "base") else voice
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(
                f"{cfg.QWEN_TTS_URL}/tts",
                json={"voice": qvoice, "text": PREVIEW_TEXT, "style": style},
            )
        if r.status_code == 200 and r.content:
            rate = int(r.headers.get("X-Sample-Rate", "24000"))
            return Response(content=_pcm_to_wav(r.content, rate), media_type="audio/wav")
    except Exception as e:  # noqa: BLE001
        log.info("voice-preview via TTS service failed (%s); falling back to static sample", e)
    # Fallback: the static reference clip.
    path = cfg.VOICE_SAMPLES.get(voice) or cfg.VOICE_SAMPLES_FALLBACK.get(voice)
    if path and path.is_file():
        return FileResponse(path, media_type="audio/wav")
    return JSONResponse(status_code=404, content={"ok": False, "error": f"no preview for '{voice}'"})


# ---------------------------------------------------------------------------
# Serve the launcher + interview pages + plans as real files (mounted LAST so the
# /api/* routes above win). "/" -> index.html, and /interview.html, /candidates.js,
# /plans/<slug>.json, /plan.json are served from launcher/. Because /interview.html
# is a real file here, it is NOT redirected into a Jitsi room.
# ---------------------------------------------------------------------------
if cfg.LAUNCHER_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(cfg.LAUNCHER_DIR), html=True), name="site")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=cfg.HOST, port=cfg.PORT)
