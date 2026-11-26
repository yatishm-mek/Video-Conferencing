"""Pipecat glue: feeds pipeline events into the controller and carries out its actions.

Sits where the LLM sits in a normal Pipecat bot (after the user aggregator,
before TTS). It turns frames into controller events:

  LLMContextFrame           -> user_turn(text)     (a complete candidate turn)
  UserStartedSpeakingFrame  -> user_started()
  InterruptionFrame         -> interruption()      (candidate cut the bot off)
  BotStarted/Stopped        -> bot_started() / bot_finished() (debounced)
  timers, evaluator results -> timer(name) / eval_done(result)

Timer and evaluator events are queued to this processor as *uninterruptible*
frames, so a barge-in can't drop them. Asking a question streams from the LLM,
so it runs from an *interruptible* frame: a barge-in cancels it mid-stream.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    DataFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from interview_bot import debug
from interview_bot.services.bot_speaker import BotVoice
from interview_bot.controller.controller import (
    Action,
    Ask,
    CancelTimer,
    Controller,
    ControllerConfig,
    Clarify,
    EndInterview,
    Evaluate,
    Rephrase,
    Respond,
    Say,
    StartTimer,
    State,
)
from interview_bot.controller.echo import last_user_text
from interview_bot.services.evaluator import Evaluator
from interview_bot.controller.speech_clock import SpeechClock
from interview_bot.controller.timers import Timers
from interview_bot.models.schemas import EvalResult, QuestionPlan


@dataclass
class BeginFrame(DataFrame):
    interruptible: bool = field(default=False, init=False)


@dataclass
class TimerFiredFrame(DataFrame):
    timer: str = ""
    interruptible: bool = field(default=False, init=False)


@dataclass
class EvalDoneFrame(DataFrame):
    result: EvalResult | None = None
    interruptible: bool = field(default=False, init=False)


@dataclass
class BotFinishedFrame(DataFrame):
    interruptible: bool = field(default=False, init=False)


@dataclass
class AskFrame(DataFrame):
    """Interruptible: a barge-in cancels the LLM stream."""

    ask: Ask | Rephrase | Clarify | Respond | None = None


# SessionStore now lives in the repository layer; re-exported here so existing
# `from interview_bot.controller.session import SessionStore` call sites keep working.
from interview_bot.repository.session_store import SessionStore  # noqa: E402


# Timers that mean "N seconds of candidate silence".
SILENCE_TIMERS = {"no_speech", "need_time_reply", "extended_wait", "incomplete", "ready_reprompt"}


def silence_gate(name: str, full_secs: float, clock: SpeechClock) -> float | None:
    """For a silence timer that just fired: None if the candidate really has
    been silent that long, else how many more seconds to wait."""
    if name not in SILENCE_TIMERS:
        return None
    if clock.user_speaking:
        return full_secs
    remaining = full_secs - clock.silence_secs()
    return remaining if remaining > 0.05 else None


@dataclass
class SessionConfig:
    resume_max_age_secs: float = 900
    bot_finished_debounce_secs: float = 0.8
    speech_watchdog_secs: float = 8.0


class InterviewSession(FrameProcessor):
    def __init__(
        self,
        plan: QuestionPlan,
        voice: BotVoice,
        evaluator: Evaluator,
        store: SessionStore,
        controller_cfg: ControllerConfig,
        cfg: SessionConfig,
        *,
        speech_clock: SpeechClock | None = None,
        echo_guard=None,
        fresh: bool = False,
        controller=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctrl = controller if controller is not None else Controller(plan, controller_cfg)
        self.speech = speech_clock or SpeechClock()
        self._voice = voice
        self._evaluator = evaluator
        self._store = store
        self._cfg = cfg
        self._timers = Timers(self._on_timer)
        self._bot_speaking = False
        self._awaiting_audio = False  # speech sent, its audio hasn't started yet
        self._utterance_started: float | None = None  # when its audio began
        self._debounce: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None
        self._eval_task: asyncio.Task | None = None
        self.resuming = False
        self._user_msgs = 0  # user messages in the context we've already handled
        self._echo = echo_guard

        saved = None if fresh else store.load()
        if saved and self._resumable(saved):
            self.ctrl.load_dict(saved)
            self.resuming = True
            logger.info(f"[session] resuming {plan.candidate} at q{self.ctrl.q_index + 1} ({saved['state']})")

    def _resumable(self, saved: dict) -> bool:
        age = time.time() - saved.get("saved_at", 0)
        return saved.get("state") not in (State.DONE, State.REPORT) and age < self._cfg.resume_max_age_secs

    # -- called from the pipeline -------------------------------------------------

    async def begin(self):
        await self.queue_frame(BeginFrame())

    async def on_disconnected(self):
        self._cancel_aux()
        await self._run(self.ctrl.disconnected())
        self._timers.cancel_all()

    # -- frames ---------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # System frames: react immediately (before super() cancels a reply).
        if isinstance(frame, InterruptionFrame):
            heard = (
                time.monotonic() - self._utterance_started
                if self._utterance_started is not None and not self._awaiting_audio
                else 0.0
            )
            # The cut-off speech will never play: forget we were waiting for it.
            self._awaiting_audio = False
            self._cancel_task("_watchdog")
            self._cancel_task("_debounce")
            await self._run(self.ctrl.interruption(heard))
            if self.ctrl.pending_turn_id is None:
                self._cancel_task("_eval_task")  # its answer was taken back: free the LLM
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._run(self.ctrl.user_started())
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            if self._awaiting_audio or self._utterance_started is None:
                self._utterance_started = time.monotonic()  # first audio of the new line
            self._awaiting_audio = False
            self._cancel_task("_debounce")
            self._cancel_task("_watchdog")
            await self._run(self.ctrl.bot_started())
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._cancel_task("_debounce")
            self._debounce = self.create_task(self._finish_after(self._cfg.bot_finished_debounce_secs))

        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            n = sum(1 for m in frame.context.get_messages() if isinstance(m, dict) and m.get("role") == "user")
            if n <= self._user_msgs:
                # Every word of this turn was dropped (noise, echo, "Thank you."
                # hallucination): the context still ends with the OLD reply.
                logger.debug("[session] turn had no new words; ignoring")
                before = self.ctrl.false_cuts
                await self._run(self.ctrl.empty_turn())
                if self._echo is not None and self.ctrl.false_cuts > before:
                    self._echo.false_cut()
            else:
                self._user_msgs = n
                text = last_user_text(frame)
                debug.trace("user_turn", text=text, silence=round(self.speech.silence_secs(), 2))
                await self._run(self.ctrl.user_turn(text, self.speech.silence_secs()))
        elif isinstance(frame, BeginFrame):
            await self._run(self.ctrl.resume() if self.resuming else self.ctrl.start())
        elif isinstance(frame, TimerFiredFrame):
            await self._run(self.ctrl.timer(frame.timer))
        elif isinstance(frame, EvalDoneFrame):
            await self._run(self.ctrl.eval_done(frame.result))
        elif isinstance(frame, BotFinishedFrame):
            await self._run(self.ctrl.bot_finished())
        elif isinstance(frame, AskFrame):
            a = frame.ask
            self._expect_speech()
            if isinstance(a, Rephrase):
                await self._voice.rephrase(self._push_voice, a.text, a.rubric)
            elif isinstance(a, Respond):
                await self._voice.respond(self._push_voice, a.said, a.line, a.rubric)
            elif isinstance(a, Clarify):
                await self._voice.clarify(
                    self._push_voice, a.question, a.asked, a.rubric,
                    context=a.context, answers=a.answers, earlier=a.earlier,
                )
            else:
                await self._voice.ask(self._push_voice, a.reply, a.question, a.number, a.previous)
        else:
            await self.push_frame(frame, direction)

    # -- actions ----------------------------------------------------------------------

    async def _run(self, actions: list[Action]):
        for a in actions:
            debug.trace(
                "action",
                kind=type(a).__name__,
                state=getattr(self.ctrl, "state", None),
                q=getattr(self.ctrl, "q_index", None),
                detail=getattr(a, "text", None) or getattr(a, "name", None) or getattr(a, "reason", ""),
            )
            if isinstance(a, Say):
                self._expect_speech()
                await self._voice.say(self._push_voice, a.text)
            elif isinstance(a, (Ask, Rephrase, Clarify, Respond)):
                await self.queue_frame(AskFrame(ask=a))
            elif isinstance(a, Evaluate):
                self._cancel_task("_eval_task")  # an older evaluation would only queue in Ollama
                self._eval_task = self.create_task(self._evaluate(a), "evaluate")
            elif isinstance(a, StartTimer):
                self._timers.start(a.name, a.secs)
            elif isinstance(a, CancelTimer):
                self._timers.cancel(a.name)
            elif isinstance(a, EndInterview):
                logger.info(f"[session] interview finished ({a.reason}); results: {self.ctrl.results}")
                if self.ctrl.flags:
                    logger.warning(f"[session] flags for human review: {self.ctrl.flags}")
        if actions:
            self._save()

    async def _push_voice(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Bot speech goes out through here, so the echo guard knows what was said."""
        if self._echo is not None and isinstance(frame, LLMTextFrame):
            self._echo.bot_said(frame.text)
        await self.push_frame(frame, direction)

    async def _evaluate(self, a: Evaluate):
        t0 = time.monotonic()
        try:
            result = await self._evaluator.evaluate(a.request)
        except Exception as e:
            logger.warning(f"[session] evaluator failed ({e}); defaulting to next")
            result = EvalResult.default_next(a.request.turn_id, a.request.answer, f"evaluator error: {e}")
        logger.info(
            f"[eval] {a.request.turn_id} {time.monotonic() - t0:.2f}s intent={result.intent} "
            f"action={result.action} complete={result.complete}"
        )
        await self.queue_frame(EvalDoneFrame(result=result))

    async def _on_timer(self, name: str):
        full = self._timers.durations.get(name, 0.0)
        wait = silence_gate(name, full, self.speech)
        if wait is not None:
            logger.debug(f"[timer] {name}: candidate not silent for {full:g}s yet; waiting {wait:.1f}s more")
            self._timers.start(name, wait, full=full)
            return
        await self.queue_frame(TimerFiredFrame(timer=name))

    # -- "bot finished speaking" detection --------------------------------------------

    async def _finish_after(self, secs: float):
        # Bot audio stops briefly between sentences; only a longer stop, with no
        # LLM text still streaming, means the bot's turn is over.
        await asyncio.sleep(secs)
        while self._voice.llm_active:
            await asyncio.sleep(0.1)
        # New speech was sent after this stop and hasn't started playing: the
        # stop belonged to the previous utterance. (Seen live: a stop left over
        # from the greeting ended question 1 before its audio began.)
        if self._awaiting_audio:
            self._debounce = None
            return
        if not self._bot_speaking:
            self._debounce = None
            await self.queue_frame(BotFinishedFrame())

    def _expect_speech(self):
        self._awaiting_audio = True
        self._cancel_task("_debounce")
        # If TTS never produces audio (error), don't wait forever for it.
        self._cancel_task("_watchdog")
        self._watchdog = self.create_task(self._watchdog_after(self._cfg.speech_watchdog_secs))

    async def _watchdog_after(self, secs: float):
        await asyncio.sleep(secs)
        self._watchdog = None
        if self._awaiting_audio and not self._bot_speaking:
            self._awaiting_audio = False
            logger.warning("[session] no bot audio after speech was sent; continuing")
            await self.queue_frame(BotFinishedFrame())

    def _cancel_task(self, attr: str):
        task = getattr(self, attr)
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
        setattr(self, attr, None)

    def _cancel_aux(self):
        for attr in ("_debounce", "_watchdog", "_eval_task"):
            self._cancel_task(attr)

    async def cleanup(self):
        self._cancel_aux()
        self._timers.cancel_all()
        await super().cleanup()

    def _save(self):
        try:
            self._store.save(self.ctrl.to_dict())
        except OSError as e:
            logger.warning(f"[session] could not save state: {e}")


# ---------------------------------------------------------------------------
# Natural engine (BOT_ENGINE=natural): one streaming LLM turn per reply,
# background scoring per finished question.
# ---------------------------------------------------------------------------

from collections import deque  # noqa: E402

from pipecat.frames.frames import LLMFullResponseEndFrame  # noqa: E402

from interview_bot.controller.natural import (  # noqa: E402
    CancelTurn,
    NaturalController,
    Score,
    StaleTurn,
    Turn,
    TurnRunner,
)


@dataclass
class TurnDoneFrame(DataFrame):
    outcome: object = None
    turn_id: str = ""
    interruptible: bool = field(default=False, init=False)


@dataclass
class ScoreDoneFrame(DataFrame):
    question_id: int = 0
    result: dict | None = None
    interruptible: bool = field(default=False, init=False)


class NaturalSession(InterviewSession):
    def __init__(self, plan, voice, runner: TurnRunner, scorer, store, controller: NaturalController, cfg, **kwargs):
        super().__init__(plan, voice, None, store, None, cfg, controller=controller, **kwargs)
        self._runner = runner
        self._scorer = scorer  # async (ScoreRequest) -> dict | None
        self._turn_task: asyncio.Task | None = None
        self._score_task: asyncio.Task | None = None
        self._score_req = None
        self._scores: deque = deque()

    async def begin(self):
        from interview_bot.controller.natural import warmup_messages

        self.create_task(
            self._runner.prewarm(warmup_messages(self.ctrl.plan, self.ctrl.cfg.time_limit_secs, self.ctrl.cfg.mode,
                                                 self.ctrl.jd_text, self.ctrl.cfg.label_format)), "prewarm"
        )
        await super().begin()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InterruptionFrame):
            self._cancel_task("_turn_task")
        if isinstance(frame, TurnDoneFrame):
            await FrameProcessor.process_frame(self, frame, direction)
            if frame.outcome is not None:
                acts = self.ctrl.turn_done(frame.outcome)
            else:
                acts = self.ctrl.turn_failed(frame.turn_id)
            if frame.outcome is not None and frame.outcome.open_response:
                # The question goes into the reply's open TTS response: one utterance,
                # so "bot finished" can't fire between the bridge and the question.
                rest = []
                for a in acts:
                    if isinstance(a, Say) and not rest:
                        await self._push_voice(LLMTextFrame(a.text + " "))
                    else:
                        rest.append(a)
                await self._push_voice(LLMFullResponseEndFrame())
                acts = rest
            self._voice.llm_active = False
            await self._run(acts)
            self._pump_scores()
            return
        if isinstance(frame, ScoreDoneFrame):
            await FrameProcessor.process_frame(self, frame, direction)
            await self._run(self.ctrl.score_done(frame.question_id, frame.result))
            self._save()
            self._pump_scores()
            return
        await super().process_frame(frame, direction)

    async def _run(self, actions):
        rest = []
        for a in actions:
            debug.trace("nat_action", kind=type(a).__name__, reason=getattr(a, "reason", ""))
            if isinstance(a, Turn):
                self._start_turn(a.request)
            elif isinstance(a, CancelTurn):
                if self._turn_task is not None:
                    logger.debug(f"[session] cancelling the reply ({a.reason})")
                self._cancel_task("_turn_task")
                self._voice.llm_active = False
            elif isinstance(a, Score):
                self._scores.append(a.request)
            else:
                rest.append(a)
        await super()._run(rest)
        self._pump_scores()
        if any(isinstance(a, EndInterview) for a in rest):
            # The closing line has played: hang up, like an interviewer ending the call.
            from pipecat.frames.frames import EndWorkerFrame

            logger.info("[session] closing line finished; ending the call")
            await self.push_frame(EndWorkerFrame(reason="interview finished"))

    def _start_turn(self, req):
        # The reply has priority: a running background score is stopped and redone later.
        if self._score_task is not None and not self._score_task.done():
            self._cancel_task("_score_task")
            if self._score_req is not None:
                self._scores.appendleft(self._score_req)
        self._cancel_task("_turn_task")
        self._voice.llm_active = True  # "bot finished" waits until the turn's result is handled
        self._turn_task = self.create_task(self._do_turn(req), "turn")

    async def _do_turn(self, req):
        try:
            outcome = await self._runner.run(req, self._push_voice, self.ctrl, self._expect_speech)
        except StaleTurn:
            self._voice.llm_active = False
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[session] turn {req.turn_id} failed: {e}")
            outcome = None
        await self.queue_frame(TurnDoneFrame(outcome=outcome, turn_id=req.turn_id))

    def _pump_scores(self):
        if self._turn_task is not None and not self._turn_task.done():
            return
        if self._score_task is not None and not self._score_task.done():
            return
        if not self._scores:
            return
        self._score_req = self._scores.popleft()
        self._score_task = self.create_task(self._do_score(self._score_req), "score")

    async def _do_score(self, req):
        result = await self._scorer(req)
        self._score_req = None
        await self.queue_frame(ScoreDoneFrame(question_id=req.question_id, result=result))

    def _cancel_aux(self):
        super()._cancel_aux()
        for attr in ("_turn_task", "_score_task"):
            self._cancel_task(attr)

    async def finish_scores(self, timeout_secs: float = 90.0):
        """The call is over but questions are still waiting to be scored (on CPU a score
        takes several seconds, and each reply pauses scoring): score them now, so the
        report never ends up with an unscored last question."""
        todo = list(self._scores)
        self._scores.clear()
        if self._score_req is not None:
            todo.insert(0, self._score_req)  # the running one is cancelled and redone here
            self._score_req = None
        self._cancel_task("_score_task")
        if not todo:
            return
        logger.info(f"[session] the call ended; scoring {len(todo)} remaining question(s)")
        for req in todo:
            try:
                result = await asyncio.wait_for(self._scorer(req), timeout_secs)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[session] could not score q{req.question_id} after the call: {e or type(e).__name__}")
                continue
            self.ctrl.score_done(req.question_id, result)
            self._save()

    async def cleanup(self):
        self._cancel_task("_turn_task")
        try:
            await self.finish_scores()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[session] finishing scores failed: {e}")
        await super().cleanup()
