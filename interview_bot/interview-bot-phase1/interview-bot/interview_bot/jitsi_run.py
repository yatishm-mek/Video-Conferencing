"""Run the interview bot AS A PARTICIPANT in a Jitsi room.

    python -m interview_bot.jitsi_run --room ezyhire-interview-01 \
        --name "AI Interviewer" --stt moonshine --tts standard --llm own

It starts the loopback WS bridge, launches a headless Chromium that joins the
Jitsi room (bridge/agent.js), then runs the same interview pipeline as the
browser server — only the transport differs (JitsiBridgeTransport instead of
SmallWebRTC). The orchestrator starts this per bot interview.

Needs the same models as the browser server (Ollama up; Whisper/Moonshine/Kokoro/
Qwen as selected) plus Playwright Chromium (`playwright install chromium`).
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import sys

from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.base_transport import TransportParams

from interview_bot.config import get_settings
from interview_bot.controller.pipeline import build_worker_for_transport, jd_text_for
from interview_bot.models.schemas import QuestionPlan
from interview_bot.services.llm import get_llm
from interview_bot.services.stt import preload_stt
from interview_bot.services.tts import preload_tts
from interview_bot.transports.jitsi_bridge import JitsiBridgeClient, JitsiBridgeTransport
from interview_bot.transports.jitsi_bridge.launcher import JitsiBrowserLauncher


def _overrides_from_args(args) -> dict:
    o: dict = {}
    if args.stt == "moonshine":
        o["STT_PROVIDER"] = "moonshine"
    elif args.stt == "whisper":
        o["STT_PROVIDER"] = "local"
    # Kokoro = fast local Pipecat TTS (does NOT touch the Qwen GPU microservice).
    if args.tts == "kokoro":
        o["TTS_PROVIDER"] = "kokoro"
    # The Qwen cloned voices. "standard" = one fixed fine-tuned voice (unified).
    elif args.tts in ("standard", "rithika", "aditi", "sharanya", "suresh"):
        o["TTS_PROVIDER"] = "qwen"
        o["QWEN_VOICE"] = "base" if args.tts == "standard" else args.tts
        if args.style in ("steady", "balanced", "expressive"):
            o["QWEN_STYLE"] = args.style
    # llm "openai" has no key yet → ignored (defaults to Ollama).
    if args.plan:
        o["INTERVIEW_PLAN_PATH"] = args.plan
    return o


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _warm(settings):
    """Load models up front, like the browser server's lifespan does."""
    if settings.BOT_MODE == "interview":
        try:
            plan = QuestionPlan.load(settings.INTERVIEW_PLAN_PATH)
            llm = get_llm(settings)
            if settings.BOT_ENGINE == "natural":
                from interview_bot.controller.natural import warmup_messages as natural_warmup

                await llm.warmup(
                    settings.LIVE_LLM_MODEL,
                    prime=natural_warmup(
                        plan, settings.INTERVIEW_TIME_LIMIT_SECS, settings.NATURAL_MODE,
                        jd_text_for(settings.INTERVIEW_PLAN_PATH), settings.LABEL_FORMAT,
                    ),
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LLM warmup skipped ({e}); first replies may be slower / verbatim.")
    await asyncio.to_thread(preload_stt, settings)
    await asyncio.to_thread(preload_tts, settings)


async def run(args):
    settings = get_settings().model_copy(update=_overrides_from_args(args))
    logger.info(
        f"Jitsi bot: room={args.room} stt={settings.STT_PROVIDER} "
        f"tts={settings.TTS_PROVIDER}/{settings.QWEN_VOICE} llm={settings.LIVE_LLM_MODEL}"
    )
    await _warm(settings)

    # The candidate's display name: the bridge captures ONLY this participant's
    # mic (not the recruiter's), so the STT gets one clean stream. Prefer the plan's
    # candidate; fall back to --candidate.
    candidate_name = args.candidate or ""
    try:
        candidate_name = QuestionPlan.load(settings.INTERVIEW_PLAN_PATH).candidate or candidate_name
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[jitsi-run] could not read candidate from plan ({e}); using --candidate={candidate_name!r}")
    logger.info(f"[jitsi-run] capturing only the candidate's mic: {candidate_name!r}")

    client = JitsiBridgeClient(ws_port=_free_port())
    launcher = JitsiBrowserLauncher(
        room=args.room, display_name=args.name, ws_port=client.ws_port, candidate_name=candidate_name
    )

    # Start the WS server BEFORE the browser so the first connection lands.
    await client.connect()
    await launcher.start()

    transport = JitsiBridgeTransport(
        client,
        TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=48000,  # Pipecat resamples Kokoro 24k → 48k for Jitsi
        ),
    )
    worker = build_worker_for_transport(transport, settings, fresh=True)
    try:
        await PipelineRunner(handle_sigint=False).run(worker)
    finally:
        await launcher.stop()
        await client.disconnect()


def main():
    p = argparse.ArgumentParser(description="Run the interview bot inside a Jitsi room.")
    p.add_argument("--room", required=True)
    p.add_argument("--name", default="AI Interviewer")
    p.add_argument("--plan", default=None, help="path to an interview plan JSON (defaults to config)")
    p.add_argument("--stt", default="moonshine", choices=["moonshine", "whisper"])
    p.add_argument("--tts", default="standard", choices=["kokoro", "standard", "rithika", "aditi", "sharanya", "suresh"])
    p.add_argument("--style", default="balanced", choices=["steady", "balanced", "expressive"])
    p.add_argument("--llm", default="own", choices=["own", "openai"])
    p.add_argument("--candidate", default=None, help="candidate name (informational)")
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    # Also write a log file so the run can be inspected afterwards (the console
    # scrolls away). One file per room, overwritten each run.
    from interview_bot.config import get_settings as _gs

    try:
        log_dir = _gs().LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_room = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.room)[:60]
        log_path = log_dir / f"jitsi_bot_{safe_room}.log"
        logger.add(str(log_path), level="DEBUG", mode="w", enqueue=True,
                   backtrace=True, diagnose=False)
        logger.info(f"[jitsi-run] logging to {log_path}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[jitsi-run] could not open log file ({e}); console only")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
