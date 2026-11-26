"""FastAPI server: serves the browser client and handles WebRTC signalling.

Run:  python -m interview_bot.server
Open: http://127.0.0.1:7860
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from loguru import logger
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from interview_bot import debug
from interview_bot.config import PROJECT_ROOT, get_settings
from interview_bot.controller.pipeline import run_session
from interview_bot.services.stt import preload_stt
from interview_bot.services.tts import preload_tts
from interview_bot.services.bot_speaker import warmup_messages
from interview_bot.services.evaluator import EVAL_SCHEMA, eval_warmup_messages
from interview_bot.services.llm import get_llm
from interview_bot.models.schemas import QuestionPlan

settings = get_settings()
logger.remove()
logger.add(sys.stderr, level=settings.LOG_LEVEL)
# Same log, also on disk, so it can be shared or inspected after a run.
logger.add(settings.LOGS_DIR / "server.log", level=settings.LOG_LEVEL, rotation="10 MB", retention=3)
# Opt-in: when DEBUG_TRACE is set, also write logs/debug.log and emit [trace] lines.
debug.install_debug_logging(settings)

WEB_DIR = PROJECT_ROOT / "interview_bot" / "web"
_handler = SmallWebRTCRequestHandler()
_sessions: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"STT provider: {settings.STT_PROVIDER}. Preloading models...")
    if settings.BOT_MODE == "interview":
        # The LLM first, then Whisper: loading the LLM after Whisper took the GPU crashed
        # Ollama's runner 3 times today (exit 0xc0000409), and the first turn then failed.
        await _load_llm()
    await asyncio.to_thread(preload_stt, settings)
    await asyncio.to_thread(preload_tts, settings)
    if settings.BOT_MODE == "interview" and settings.BOT_ENGINE == "natural":
        try:
            from interview_bot.controller.pipeline import FIXED_LINES
            from interview_bot.services.tts import preload_lines

            plan = QuestionPlan.load(settings.INTERVIEW_PLAN_PATH)
            await asyncio.to_thread(preload_lines, settings, [q.q for q in plan.questions] + FIXED_LINES)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[tts] could not voice the lines in advance ({e}); they are voiced live")
    logger.info(f"Ready on http://{settings.HOST}:{settings.PORT}  (mode: {settings.BOT_MODE})")
    yield
    for t in list(_sessions):
        t.cancel()
    await _handler.close()
    await get_llm(settings).close()


async def _load_llm():
    for attempt in (1, 2):
        try:
            await _warm_llm()
            return
        except Exception as e:
            if attempt == 1:
                logger.warning(f"Loading {settings.LIVE_LLM_MODEL} failed ({e}); trying once more in 5 s")
                await asyncio.sleep(5)
                continue
            logger.error(
                f"Could not reach Ollama at {settings.OLLAMA_BASE_URL} or load "
                f"{settings.LIVE_LLM_MODEL} ({e}). Is Ollama running? Until it is, the bot "
                "asks each question word for word."
            )


async def _warm_llm():
    plan = QuestionPlan.load(settings.INTERVIEW_PLAN_PATH)
    llm = get_llm(settings)
    if settings.BOT_ENGINE == "natural":
        from interview_bot.controller.natural import warmup_messages as natural_warmup
        from interview_bot.controller.pipeline import jd_text_for

        # Same system prompt for every turn and score call: cached once here.
        await llm.warmup(
            settings.LIVE_LLM_MODEL,
            prime=natural_warmup(plan, settings.INTERVIEW_TIME_LIMIT_SECS, settings.NATURAL_MODE,
                                 jd_text_for(settings.INTERVIEW_PLAN_PATH), settings.LABEL_FORMAT),
        )
    else:
        await llm.warmup(settings.LIVE_LLM_MODEL, prime=warmup_messages(plan))
    if settings.EVALUATOR == "llm" and settings.BOT_ENGINE == "classic":
        # Warm the evaluator's prompt + JSON path too (first one is slow).
        t0 = asyncio.get_running_loop().time()
        await llm.chat(
            settings.LIVE_LLM_MODEL,
            eval_warmup_messages(plan.questions[0], plan.stt_vocab()),
            max_tokens=40,
            format=EVAL_SCHEMA,
        )
        logger.info(f"Evaluator warmed up in {asyncio.get_running_loop().time() - t0:.1f}s")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


def session_settings_from(request_data: dict):
    """Build a per-session Settings copy from launcher choices (stt/llm/tts).

    get_settings() is lru-cached and shared, so we never mutate it — model_copy
    gives an independent instance that every downstream builder already reads from.
    Unknown/unavailable choices (openai, arabic) are ignored and fall back to the
    configured defaults.
    """
    if not isinstance(request_data, dict):
        return settings
    overrides: dict = {}

    stt = request_data.get("stt")
    if stt == "moonshine":
        overrides["STT_PROVIDER"] = "moonshine"
    elif stt == "whisper":
        overrides["STT_PROVIDER"] = "local"  # faster-whisper via Pipecat

    tts = request_data.get("tts")
    if tts in ("standard", "rithika", "aditi", "sharanya", "suresh"):
        # All voices are Qwen3-TTS now (Kokoro removed). "standard" = the base model.
        overrides["TTS_PROVIDER"] = "qwen"
        overrides["QWEN_VOICE"] = "base" if tts == "standard" else tts
    style = request_data.get("style")
    if style in ("steady", "balanced", "expressive"):
        overrides["QWEN_STYLE"] = style

    # llm: "own" is the default; "openai" has no key yet, so it is ignored.

    if not overrides:
        return settings
    logger.info(f"Per-session overrides from launcher: {overrides}")
    return settings.model_copy(update=overrides)


@app.post("/api/offer")
async def offer(request: Request):
    body = await request.json()
    request_data = body.get("requestData") or body.get("request_data") or {}
    fresh = bool(request_data.get("fresh")) if isinstance(request_data, dict) else False
    session_settings = session_settings_from(request_data)

    async def on_connection(connection: SmallWebRTCConnection):
        task = asyncio.create_task(run_session(connection, session_settings, fresh))
        _sessions.add(task)
        task.add_done_callback(_sessions.discard)

    return await _handler.handle_web_request(SmallWebRTCRequest.from_dict(body), on_connection)


@app.patch("/api/offer")
async def ice_candidates(request: Request):
    body = await request.json()
    await _handler.handle_patch_request(
        SmallWebRTCPatchRequest(
            pc_id=body["pc_id"],
            candidates=[
                IceCandidate(c["candidate"], c.get("sdp_mid"), c.get("sdp_mline_index"))
                for c in body.get("candidates", [])
            ],
        )
    )
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
