"""Pipecat pipeline wiring for one interview session (one WebRTC connection).

  browser mic -> transport.input
              -> STT (faster-whisper | Groq)        transcribes VAD-bounded segments
              -> HallucinationFilter                drops "thank you for watching", loops, etc.
              -> user aggregator                    Silero VAD + barge-in policy + Smart Turn v3
              -> InterviewSession                   controller + timers; Qwen phrases questions
                                                    (EchoResponder with BOT_MODE=echo)
              -> Kokoro TTS                         streams audio sentence by sentence
              -> transport.output -> browser speaker
              -> assistant aggregator               records what the bot actually said

VAD runs inside the user aggregator and broadcasts its start/stop frames
upstream to the STT, which is how Pipecat 1.x drives segmented STT.
"""

from __future__ import annotations

from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import StrategyMode, UserTurnStrategies, UserTurnStrategiesParams

from interview_bot.config import Settings
from interview_bot.services.bot_speaker import BotVoice, SpeakerConfig
from interview_bot.controller.controller import ControllerConfig
from interview_bot.controller.diagnostics import AudioLevelLogger
from interview_bot.controller.echo import EchoResponder
from interview_bot.services.evaluator import EvalConfig, LLMEvaluator, RuleEvaluator
from interview_bot.controller.natural import NaturalConfig, NaturalController, TurnRunner, run_score
from interview_bot.controller.decide import DecideRunner
from interview_bot.controller.two_step import TwoStepRunner
from interview_bot.controller.session import InterviewSession, NaturalSession, SessionConfig, SessionStore
from interview_bot.controller.echo_guard import EchoFilter, EchoGuard
from interview_bot.controller.speech_clock import PlayedTextTap, SpeechClock, SpeechClockTap
from interview_bot.services.stt import HallucinationFilter, build_stt, noisy_room
from interview_bot.services.tts import build_tts

FIXED_LINES = [
    "Sure.", "Alright, let's begin.", "Okay, thank you.", "Okay, let's move on.", "Alright, let's continue.",
    "Sure, let's go back.", "Okay. This is the last time I'll repeat it.", "That was my last question.",
    "Is there anything you'd like to add before we finish?",
    "Would you like to add anything more before we move to the next question?",
    "Are you ready to begin?", "Thank you.", "Sure. Please listen carefully this time.",
    "Thank you, I've noted that for the previous question.",
    "Would you like a moment to think?",
    "Are you there? If yes, please respond, or else I'll move to the next question.",
    "Since I haven't heard from you, let's move on to the next question.",
    "Why didn't you respond?", "Okay.", "This is an interview with a time limit, so please don't let that happen again.",
    "Please go on with your answer.", "Let's move on.",
    "Just to confirm, would you like to add anything more before we move on?",
    "Just to confirm, is there anything you'd like to add before we finish?",
    "Just to confirm: should I end the interview now? Please say yes or no.",
]
from interview_bot.controller.turn_taking import (
    DEFAULT_BACKCHANNELS,
    BargeInStartStrategy,
    ShortReplyStopStrategy,
    parse_backchannels,
)
from interview_bot.services.llm import get_llm
from interview_bot.models.schemas import QuestionPlan

ECHO_GREETING = (
    "Hi, I'm the echo bot. Say something and I'll repeat it. "
    "Say long test if you want to practise interrupting me."
)


def natural_config(s: Settings) -> NaturalConfig:
    return NaturalConfig(
        model=s.LIVE_LLM_MODEL,
        mode=s.NATURAL_MODE,
        label_format=s.LABEL_FORMAT,
        temperature=s.NATURAL_TEMPERATURE,
        max_tokens=s.NATURAL_MAX_TOKENS,
        rescue_secs=s.NATURAL_RESCUE_SECS,
        rescue_every_turns=s.NATURAL_RESCUE_EVERY_TURNS,
        turn_timeout_secs=s.NATURAL_TURN_TIMEOUT_SECS,
        silence_prompt_secs=s.NATURAL_SILENCE_PROMPT_SECS,
        silence_move_on_secs=s.NATURAL_SILENCE_MOVE_ON_SECS,
        max_answer_secs=s.MAX_ANSWER_SECS,
        time_limit_secs=s.INTERVIEW_TIME_LIMIT_SECS,
        max_repeats=s.MAX_REPEATS_PER_QUESTION,
        max_go_backs=s.MAX_GO_BACKS_PER_INTERVIEW,
        max_confirmations=s.MAX_CONFIRMATIONS_PER_INTERVIEW,
        max_idle_turns=s.MAX_IDLE_TURNS_PER_QUESTION,
        max_end_checks=s.MAX_END_CHECKS_PER_INTERVIEW,
        max_clarifications=s.MAX_REPHRASES_PER_INTERVIEW,
        answer_hold_secs=s.ANSWER_HOLD_SECS,
        hold_min_words=s.HOLD_MIN_WORDS,
        earlier_questions=s.EVAL_HISTORY_QUESTIONS,
        early_answer_fraction=s.EARLY_ANSWER_FRACTION,
        score_temperature=s.EVAL_TEMPERATURE,
        score_max_tokens=s.SCORE_MAX_TOKENS,
        track_played=True,
    )


def jd_text_for(plan_path) -> str:
    """jd.txt next to the plan: the only role facts the bot may share with the candidate."""
    from pathlib import Path

    try:
        return (Path(plan_path).parent / "jd.txt").read_text(encoding="utf-8")
    except OSError:
        return ""


def resume_text_for(plan_path) -> str:
    """resume.txt next to the plan, for the "reading from the resume" flag."""
    from pathlib import Path

    f = Path(plan_path).parent / "resume.txt"
    try:
        return f.read_text(encoding="utf-8")
    except OSError:
        return ""


def controller_config(s: Settings) -> ControllerConfig:
    return ControllerConfig(
        no_speech_prompt_secs=s.NO_SPEECH_PROMPT_SECS,
        need_time_reply_secs=s.NEED_TIME_REPLY_SECS,
        extended_wait_secs=s.EXTENDED_WAIT_SECS,
        incomplete_prompt_secs=s.INCOMPLETE_PROMPT_SECS,
        evaluator_filler_secs=s.EVALUATOR_FILLER_SECS,
        evaluator_timeout_secs=s.EVALUATOR_TIMEOUT_SECS,
        max_answer_secs=s.MAX_ANSWER_SECS,
        max_confirmations=s.MAX_CONFIRMATIONS_PER_INTERVIEW,
        max_verifications=s.MAX_VERIFICATIONS_PER_INTERVIEW,
        history_questions=s.EVAL_HISTORY_QUESTIONS,
        history_words=s.EVAL_HISTORY_WORDS,
        time_limit_secs=s.INTERVIEW_TIME_LIMIT_SECS,
        ready_reprompt_secs=s.READY_REPROMPT_SECS,
        answer_hold_secs=s.ANSWER_HOLD_SECS,
        hold_min_words=s.HOLD_MIN_WORDS,
        backchannels=s.BACKCHANNEL_WORDS or DEFAULT_BACKCHANNELS,
        max_rephrases=s.MAX_REPHRASES_PER_INTERVIEW,
        repeats_before_best_attempt=s.REPEATS_BEFORE_BEST_ATTEMPT,
        flag_repeats_total=s.FLAG_REPEATS_TOTAL,
        flag_go_back_requests=s.FLAG_GO_BACK_REQUESTS,
        # Turn ended by Smart Turn's timeout (not by its "done" call) -> ask.
        done_check_silence_secs=s.DONE_CHECK_SILENCE_SECS,
        early_answer_fraction=s.EARLY_ANSWER_FRACTION,
    )


def build_worker(
    connection: SmallWebRTCConnection, settings: Settings, fresh: bool = False
) -> PipelineWorker:
    """Browser (SmallWebRTC) path — a candidate talks 1:1 with the bot in a page.

    The bot-in-Jitsi path (interview_bot.jitsi_run) builds a JitsiBridgeTransport
    and calls build_worker_for_transport directly; everything else is shared.
    """
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    return build_worker_for_transport(transport, settings, fresh)


def build_worker_for_transport(
    transport, settings: Settings, fresh: bool = False
) -> PipelineWorker:
    """Build the interview pipeline around any Pipecat transport (SmallWebRTC or
    the Jitsi bridge). The transport only needs .input()/.output() and the
    on_client_connected / on_client_disconnected event handlers."""

    session: InterviewSession | None = None
    speech_clock = SpeechClock()
    echo_guard = EchoGuard()
    if settings.BOT_MODE == "interview":
        plan = QuestionPlan.load(settings.INTERVIEW_PLAN_PATH)
        vocab = plan.stt_vocab()  # candidate's name + tech terms bias Whisper
        voice = BotVoice(
            plan,
            get_llm(settings),
            SpeakerConfig(
                model=settings.LIVE_LLM_MODEL,
                temperature=settings.LIVE_LLM_TEMPERATURE,
                max_tokens=settings.LIVE_LLM_MAX_TOKENS,
                first_token_timeout=settings.LLM_FIRST_TOKEN_TIMEOUT_SECS,
            ),
        )
        session_cfg = SessionConfig(
            resume_max_age_secs=settings.RESUME_MAX_AGE_SECS,
            bot_finished_debounce_secs=settings.BOT_FINISHED_DEBOUNCE_SECS,
            speech_watchdog_secs=settings.SPEECH_WATCHDOG_SECS,
        )
        if settings.BOT_ENGINE == "natural":
            ncfg = natural_config(settings)
            llm = get_llm(settings)

            async def scorer(req, _llm=llm, _cfg=ncfg):
                return await run_score(_llm, _cfg, req)

            session = NaturalSession(
                plan,
                voice,
                (DecideRunner(llm, ncfg) if ncfg.mode == "decide" else
                 TwoStepRunner(llm, ncfg) if ncfg.mode == "two_step" else TurnRunner(llm, ncfg)),
                scorer,
                SessionStore(settings.DATA_DIR, plan.candidate + " natural"),
                NaturalController(plan, ncfg, resume_text=resume_text_for(settings.INTERVIEW_PLAN_PATH),
                                  jd_text=jd_text_for(settings.INTERVIEW_PLAN_PATH)),
                session_cfg,
                speech_clock=speech_clock,
                echo_guard=echo_guard,
                fresh=fresh,
            )
        else:
            session = InterviewSession(
                plan,
                voice,
                (
                    LLMEvaluator(
                        get_llm(settings),
                        EvalConfig(
                            model=settings.LIVE_LLM_MODEL,  # same model as the bot: one LLM loaded
                            temperature=settings.EVAL_TEMPERATURE,
                            max_tokens=settings.EVAL_MAX_TOKENS,
                        ),
                    )
                    if settings.EVALUATOR == "llm"
                    else RuleEvaluator()
                ),
                SessionStore(settings.DATA_DIR, plan.candidate),
                controller_config(settings),
                SessionConfig(
                    resume_max_age_secs=settings.RESUME_MAX_AGE_SECS,
                    bot_finished_debounce_secs=settings.BOT_FINISHED_DEBOUNCE_SECS,
                    speech_watchdog_secs=settings.SPEECH_WATCHDOG_SECS,
                ),
                speech_clock=speech_clock,
                echo_guard=echo_guard,
                fresh=fresh,
            )
        bot = session
        logger.info(
            f"Interview ({settings.BOT_ENGINE} engine): {plan.candidate}, {len(plan.questions)} questions, "
            f"LLM {settings.LIVE_LLM_MODEL}"
            f"{', resuming' if session.resuming else ''}"
        )
    else:
        vocab = settings.stt_vocab_list
        bot = EchoResponder()

    stt = build_stt(settings, vocab=vocab)
    tts = build_tts(settings)

    vad = SileroVADAnalyzer(
        params=VADParams(
            confidence=settings.VAD_CONFIDENCE,
            start_secs=settings.VAD_START_SECS,
            stop_secs=settings.VAD_STOP_SECS,
            min_volume=settings.VAD_MIN_VOLUME,
        )
    )
    strategies = UserTurnStrategies(
        start=[
            BargeInStartStrategy(
                barge_in_min_secs=settings.BARGE_IN_MIN_SECS,
                barge_in_hard_secs=settings.BARGE_IN_HARD_SECS,
                early_window_secs=settings.EARLY_BARGE_IN_WINDOW_SECS,
                vad_stop_secs=settings.VAD_STOP_SECS,
                transcript_wait_secs=settings.STT_TTFS_P99_SECS,
                backchannels=parse_backchannels(settings.BACKCHANNEL_WORDS or DEFAULT_BACKCHANNELS),
                echo_guard=echo_guard,
                expects_answer=lambda: bool(getattr(getattr(bot, "ctrl", None), "expects_yes_no", lambda: False)()),
                reply_pending=lambda: bool(getattr(getattr(bot, "ctrl", None), "pending_turn_id", None) is not None
                                           or getattr(bot, "_awaiting_audio", False)),
                noisy=noisy_room,
            )
        ],
        # FIRST: either one ends the turn. A short "Yes"/"No" ends it at once.
        stop=[ShortReplyStopStrategy(), TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())],
    )

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad,
            user_turn_strategies=strategies,
            user_turn_strategies_params=UserTurnStrategiesParams(stop_mode=StrategyMode.FIRST),
            user_turn_stop_timeout=settings.USER_TURN_STOP_TIMEOUT_SECS,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            AudioLevelLogger(every_secs=settings.AUDIO_LEVEL_LOG_SECS),
            stt,
            HallucinationFilter(settings.hallucination_phrases, vocab=vocab),
            EchoFilter(echo_guard),  # the bot's own voice coming back through the mic
            SpeechClockTap(speech_clock),  # sees VAD frames on their way upstream
            aggregators.user(),
            bot,
            tts,
            transport.output(),
            PlayedTextTap(lambda text: getattr(getattr(bot, "ctrl", None), "sentence_played", lambda t: None)(text)),
            aggregators.assistant(),
        ]
    )

    latency = UserBotLatencyObserver()

    @latency.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(_obs, breakdown):
        for line in breakdown.turn_contribution_lines():
            logger.info(f"[latency] {line}")

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
        ),
        observers=[latency],
        enable_rtvi=False,  # our client is plain WebRTC, not an RTVI client
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("Client connected")
        if session:
            await session.begin()
        else:
            await worker.queue_frames([TTSSpeakFrame(ECHO_GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("Client disconnected")
        if session:
            # Saved as RECONNECTING: rejoining within RESUME_MAX_AGE_SECS
            # continues at the same question.
            await session.on_disconnected()
        await worker.cancel()

    return worker


async def run_session(connection: SmallWebRTCConnection, settings: Settings, fresh: bool = False) -> None:
    worker = build_worker(connection, settings, fresh)
    await PipelineRunner(handle_sigint=False).run(worker)
