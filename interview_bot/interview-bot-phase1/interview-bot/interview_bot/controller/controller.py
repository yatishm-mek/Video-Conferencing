"""Interview controller: the plain-Python state machine that decides everything.

It knows nothing about Pipecat, audio or LLM APIs. Events go in (the bot
finished speaking, a turn ended, a timer fired, an evaluation came back), and
a list of actions comes out (say this, ask that, start/cancel a timer,
evaluate, end). LLMs only ever *phrase* a question or *return JSON*; no LLM
decides a transition.

States: GREETING, READY_CHECK, ASK_QUESTION, LISTENING, EVALUATING, FOLLOW_UP,
CLARIFY, ANYTHING_TO_ADD, NEED_TIME, EXTENDED_WAIT, RECONNECTING, CLOSING,
REPORT, DONE.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from loguru import logger

from interview_bot.controller import intents
from interview_bot.services.evaluator import EvalRequest
from interview_bot.controller.turn_taking import DEFAULT_BACKCHANNELS, is_backchannel, parse_backchannels
from interview_bot.models.schemas import Correction, EvalResult, Question, QuestionPlan


class State(StrEnum):
    GREETING = "GREETING"
    READY_CHECK = "READY_CHECK"
    ASK_QUESTION = "ASK_QUESTION"
    LISTENING = "LISTENING"
    EVALUATING = "EVALUATING"
    FOLLOW_UP = "FOLLOW_UP"
    CLARIFY = "CLARIFY"
    ANYTHING_TO_ADD = "ANYTHING_TO_ADD"
    NEED_TIME = "NEED_TIME"
    EXTENDED_WAIT = "EXTENDED_WAIT"
    RECONNECTING = "RECONNECTING"
    CLOSING = "CLOSING"
    REPORT = "REPORT"
    DONE = "DONE"


# States in which the bot is delivering a question/prompt; when it finishes
# speaking, the controller starts listening.
_PROMPT_STATES = {State.ASK_QUESTION, State.FOLLOW_UP, State.CLARIFY, State.ANYTHING_TO_ADD}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@dataclass
class Say:
    """Speak fixed text (no LLM)."""

    text: str


@dataclass
class Ask:
    """Ask the next planned question; the bot LLM phrases a short opener + it."""

    question: Question
    number: int
    reply: str
    previous: Question | None


@dataclass
class Rephrase:
    """Say the question again in different words (LLM, guarded against hints)."""

    text: str
    rubric: list[str]
    number: int


@dataclass
class Clarify:
    """Answer the candidate's question about the question (LLM, no hints)."""

    question: str
    asked: str
    rubric: list[str]
    context: str = ""
    answers: list[str] = field(default_factory=list)
    earlier: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Respond:
    """Reply naturally to something that isn't an answer, then repeat the question."""

    said: str
    line: str
    rubric: list[str]


@dataclass
class Evaluate:
    request: EvalRequest


@dataclass
class StartTimer:
    name: str
    secs: float


@dataclass
class CancelTimer:
    name: str


@dataclass
class EndInterview:
    reason: str


Action = Say | Ask | Rephrase | Clarify | Respond | Evaluate | StartTimer | CancelTimer | EndInterview

# Every timer the controller uses. Cancelled together when the candidate speaks.
LISTEN_TIMERS = ("no_speech", "need_time_reply", "extended_wait", "ready_reprompt", "incomplete", "answer_hold")
EVAL_TIMERS = ("eval_filler", "eval_timeout")


@dataclass
class ControllerConfig:
    no_speech_prompt_secs: float = 5
    need_time_reply_secs: float = 6
    extended_wait_secs: float = 25
    incomplete_prompt_secs: float = 5
    evaluator_filler_secs: float = 0.8
    evaluator_timeout_secs: float = 3
    max_answer_secs: float = 180
    max_confirmations: int = 3
    max_verifications: int = 3
    max_reverts_per_line: int = 1  # "they were still answering" at most once per bot line (echo looped it)
    filler_min_words: int = 15  # "Okay." while evaluating only after a real answer
    filler_every: int = 3  # ... and at most once every this many answers  # re-checking a claim: 1 per question, this many per interview
    history_questions: int = 4  # earlier Q/A pairs the evaluator and clarifier see
    history_words: int = 60  # per earlier answer (keeps the prompt small on CPU)
    time_limit_secs: float = 20 * 60
    ready_reprompt_secs: float = 12
    # Wait this long after a *long* answer before moving on, so a thinking
    # pause doesn't get the next question. Short replies get no hold.
    answer_hold_secs: float = 1.2
    hold_min_words: int = 12
    backchannels: str = DEFAULT_BACKCHANNELS
    # Clarification limits (stop fishing for hints; flag for a human).
    # 1st "I didn't get it" -> verbatim, 2nd -> reworded (max per interview),
    # Nth -> "give it your best attempt" + flag.
    max_rephrases: int = 3
    repeats_before_best_attempt: int = 3
    flag_repeats_total: int = 5
    # Asking to go back to an earlier question: declined; flagged from this count.
    flag_go_back_requests: int = 2
    # A turn that ended only because Smart Turn timed out (candidate trailed
    # off, ~5 s of silence) gets "Is that your complete answer?" (once per question).
    done_check_silence_secs: float = 4.5
    # Cut in after hearing this share of a new question -> it's an early
    # answer to that question; before it -> still answering the last one.
    early_answer_fraction: float = 0.6
    speech_words_per_sec: float = 2.5


# ---------------------------------------------------------------------------
# Fixed lines
# ---------------------------------------------------------------------------

LINES = {
    "ready_wait": "Sure, take your time. Just say yes when you're ready to begin.",
    "ready_reprompt": "Whenever you're ready, just say yes and we'll begin.",
    "need_time": "Would you like a moment to think?",
    "take_time": "Sure, take your time.",
    "go_ahead": "Okay, go ahead whenever you're ready.",
    "dont_know": "That's fine, thanks for being honest.",
    "hint": "I can't give hints, but take your time.",
    "move_on": "No problem, let's move on.",
    "anything_to_add": "Is there anything you'd like to add?",
    "max_answer": "Thanks, that's very helpful. In the interest of time, let's move on.",
    "filler": "Okay, got it.",
    "fillers": ["Okay.", "Alright.", "I see.", "Right, okay."],
    "hear_yes": "Yes, I can hear you clearly.",
    "hear_go_on": "Yes, I can hear you. Please go on.",
    "skip": "Alright, let's move on to the next one.",
    "refuse": "I'd encourage you to give it a try. Even a short answer helps. Or shall we skip this one?",
    "end_confirm": "Are you sure you want to end the interview now?",
    "end_ok": "Okay, we'll end the interview here.",
    "end_cancel": "Good, let's continue.",
    "sorry": "Sorry about that.",
    "own_words": (
        "Please answer on your own, without any outside help or AI tools. "
        "I want to hear your own understanding."
    ),
    "not_heard_yet": "I haven't heard an answer from you yet.",
    "identity": "I'm an AI interviewer. A human recruiter will review this interview and make the decision.",
    "salary": "I can't discuss salary or compensation here. The recruiter will go over that with you.",
    "doubt": "Sure, what's your question?",
    "cant_answer": (
        "I'm not able to answer questions during the interview, but the recruiter can help "
        "with that afterwards. Would you like me to repeat the question?"
    ),
    "continue": "Please go ahead with your answer whenever you're ready.",
    "stay_on_topic": "That doesn't sound like an answer to my question. Let's stay focused.",
    "reconnected": "Looks like we lost connection, let's continue.",
    "rephrase_intro": "Let me put it another way.",
    "best_attempt": (
        "I can only repeat it a couple of times. Please give it your best attempt, "
        "or say skip if you'd like to move on."
    ),
    "no_going_back": (
        "Sorry, we've already moved past that question, so I can't go back to it. "
        "Let's continue with this one."
    ),
    "done_check": "Is that your complete answer?",
    "go_on": "Sure, go ahead.",
    "ended": "The interview has ended. Thank you, you can close this window now.",
}


class Controller:
    def __init__(
        self,
        plan: QuestionPlan,
        cfg: ControllerConfig | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.plan = plan
        self.cfg = cfg or ControllerConfig()
        self.clock = clock
        self._backchannels = parse_backchannels(self.cfg.backchannels)

        self.state = State.GREETING
        self.q_index = -1
        self.followups = 0
        self.confirmations_q = 0
        self.confirmations_total = 0
        self.verifications_total = 0
        self.verified_q = False
        self.past: list[tuple[str, str]] = []  # finished questions: (question, candidate's answers)
        self.history: list[tuple[str, str]] = []  # this question: ("bot"|"candidate", text)
        self.answer_parts: list[str] = []
        self.started_at: float | None = None
        self.answer_started = False
        self.need_time_used = False
        self.ready_reprompted = False
        self.force_move_on = False
        self.awaiting: str | None = None  # candidate_question | repeat_confirm | anything_to_add
        self.turn_seq = 0
        self.pending_turn_id: str | None = None
        self.pending_answer = ""
        self.held_result: EvalResult | None = None  # waiting on clarify / anything-to-add
        self.clarify_term: str | None = None
        self.clarify_heard: str | None = None
        self._undo: tuple | None = None  # state before a cut-off, in case it was only noise
        self._reverted_lines: set[str] = set()
        self.false_cuts = 0  # cut-offs that had no words (echo / noise)
        self.quiet = False  # the candidate complained: no more fillers
        self._last_filler, self._last_filler_turn = "", -99
        self.outside_help_q = 0
        self.off_topic_total = 0
        self.results: list[dict] = []  # one summary per finished question
        self.delivering = False  # bot speech issued, not finished yet
        self._silence = 0.0
        self._delivering_line = ""
        self._filler_said = False
        self._filler_n = 0
        self._just_reverted = False
        # Clarification / revisit bookkeeping and human-review flags.
        self.repeat_requests_q = 0
        self.repeat_requests_total = 0
        self.rephrases_total = 0
        self.go_back_requests = 0
        self.done_checked = False  # asked "Is that your complete answer?" for this question
        self.flags: list[dict] = []
        self._revert: dict | None = None
        self._acts: list[Action] = []

    # ------------------------------------------------------------------ util

    @property
    def question(self) -> Question | None:
        qs = self.plan.questions
        return qs[self.q_index] if 0 <= self.q_index < len(qs) else None

    def _go(self, state: State):
        if state != self.state:
            logger.info(f"[ctrl] {self.state} -> {state} (q{self.q_index + 1})")
        self.state = state

    def _out(self) -> list[Action]:
        acts, self._acts = self._acts, []
        return acts

    def _say(self, text: str):
        self._acts.append(Say(text))
        self.delivering = True
        self._delivering_line = text

    def _flag(self, kind: str, detail: str = ""):
        q = self.question
        entry = {"question_id": q.id if q else None, "flag": kind, "detail": detail, "t": self.clock()}
        self.flags.append(entry)
        logger.warning(f"[flag] q{entry['question_id']}: {kind} {detail}")

    def _timer(self, name: str, secs: float):
        if secs > 0:
            self._acts.append(StartTimer(name, secs))

    def _cancel(self, *names: str):
        self._acts.extend(CancelTimer(n) for n in names)

    def _snapshot(self) -> dict:
        keys = (
            "state", "q_index", "followups", "confirmations_q", "history", "answer_parts",
            "answer_started", "need_time_used", "force_move_on", "awaiting", "held_result",
            "results", "repeat_requests_q", "done_checked",
        )
        return {k: copy.deepcopy(getattr(self, k)) for k in keys}

    def _restore(self, snap: dict):
        for k, v in snap.items():
            setattr(self, k, v)

    def _last_bot_line(self) -> str:
        for who, text in reversed(self.history):
            if who == "bot":
                return text
        return self.question.q if self.question else ""

    def _earlier(self) -> list[tuple[str, str]]:
        out = []
        for q, a in self.past[-self.cfg.history_questions :]:
            words = a.split()
            clipped = " ".join(words[: self.cfg.history_words]) + (" ..." if len(words) > self.cfg.history_words else "")
            out.append((q, clipped or "(no answer)"))
        return out

    def _record(self, status: str, result: EvalResult | None = None):
        q = self.question
        if q is None:
            return
        self.past.append((q.q, " ".join(p for w, p in self.history if w == "candidate")))
        self.results.append(
            {
                "question_id": q.id,
                "status": status,  # answered | unanswered | dont_know | skipped_time
                "answer": " ".join(p for w, p in self.history if w == "candidate"),
                "followups": self.followups,
                "verdict": result.verdict if result else None,
                "score": result.score if result else None,
                "notes": result.notes if result else "",
                "rubric_hits": result.rubric_hits if result else [],
                "interpreted_answer": result.interpreted_answer if result else "",
                "corrections": [(c.from_, c.to) for c in result.corrections] if result else [],
                "unclear_terms": [t.model_dump() for t in result.unclear_terms] if result else [],
            }
        )

    # ---------------------------------------------------------------- events

    def start(self) -> list[Action]:
        self._go(State.GREETING)
        self._say(self._greeting())
        self._go(State.READY_CHECK)
        return self._out()

    def resume(self) -> list[Action]:
        """A new connection for an interview that was in progress."""
        if self.state in (State.DONE, State.REPORT):
            self._say(LINES["ended"])
            return self._out()
        if self.q_index < 0:
            self._go(State.READY_CHECK)
            self._say("Welcome back. " + self._greeting())
            return self._out()
        self.answer_parts, self.answer_started, self.awaiting = [], False, None
        self.pending_turn_id = None
        self._go(State.ASK_QUESTION)
        self._say(f"{LINES['reconnected']} {self._last_bot_line()}")
        return self._out()

    def disconnected(self) -> list[Action]:
        if self.state not in (State.CLOSING, State.REPORT, State.DONE):
            self._cancel(*LISTEN_TIMERS, *EVAL_TIMERS, "max_answer")
            self.delivering, self._revert, self.pending_turn_id = False, None, None
            self._go(State.RECONNECTING)
        return self._out()

    def bot_finished(self) -> list[Action]:
        """All queued bot speech has finished playing."""
        self.delivering = False
        s = self.state
        if s == State.EVALUATING:
            return self._out()  # the "Okay, got it." filler ended; keep waiting
        if s == State.ANYTHING_TO_ADD and self.awaiting != "anything_to_add":
            return self._out()  # waiting out the silence before asking
        self._revert = None
        if s == State.READY_CHECK and not self.ready_reprompted:
            self._timer("ready_reprompt", self.cfg.ready_reprompt_secs)
        elif s in _PROMPT_STATES:
            self._go(State.LISTENING if s != State.CLARIFY else State.CLARIFY)
            self._timer("no_speech", self.cfg.no_speech_prompt_secs)
        elif s == State.LISTENING:
            self._timer("no_speech", self.cfg.no_speech_prompt_secs)
        elif s == State.NEED_TIME:
            self._timer("need_time_reply", self.cfg.need_time_reply_secs)
        elif s == State.EXTENDED_WAIT:
            self._timer("extended_wait", self.cfg.extended_wait_secs)
        elif s == State.CLOSING:
            self._go(State.REPORT)
            # Phase 6 writes the recruiter report here.
            self._go(State.DONE)
            self._acts.append(EndInterview("completed"))
        return self._out()

    def bot_started(self) -> list[Action]:
        """Bot audio started. If a gap between two sentences looked like the
        end of the bot's turn, this undoes the silence timers it started."""
        self.delivering = True
        self._cancel("no_speech", "need_time_reply", "ready_reprompt")
        return self._out()

    def user_started(self) -> list[Action]:
        """The candidate started speaking (a user turn began)."""
        self._cancel(*LISTEN_TIMERS)
        if self.state == State.EXTENDED_WAIT:
            self._go(State.LISTENING)
        if self.state == State.EVALUATING:
            # They're still answering: drop the evaluation in flight.
            self.pending_turn_id = None
            self._cancel(*EVAL_TIMERS)
            self._go(State.LISTENING)
        if self.state == State.ANYTHING_TO_ADD and not self.delivering:
            self._go(State.LISTENING)
        if self.state == State.LISTENING and not self.answer_started:
            self.answer_started = True
            self._timer("max_answer", self.cfg.max_answer_secs)
        return self._out()

    def interruption(self, heard_secs: float = 0.0) -> list[Action]:
        """The candidate cut the bot off (the transport already stopped audio).

        `heard_secs`: how much of the bot's current line had played. Cut in
        early -> they're still on the previous question (go back). Cut in after
        most of a *new question* -> they're answering it early (stay).
        """
        words = len(self._delivering_line.split())
        est_secs = max(1.0, words / self.cfg.speech_words_per_sec)
        heard_most = heard_secs / est_secs >= self.cfg.early_answer_fraction
        can_revert = self._delivering_line not in self._reverted_lines
        if (
            self.delivering
            and self._revert is not None
            and heard_most
            and self.state in (State.ASK_QUESTION, State.FOLLOW_UP)
        ):
            logger.info(
                f"[ctrl] cut in after {heard_secs:.1f}s of ~{est_secs:.1f}s: early answer to q{self.q_index + 1}"
            )
            self._go(State.LISTENING)
        elif self.delivering and self._revert is not None and can_revert:
            self._reverted_lines.add(self._delivering_line)
            logger.info(f"[ctrl] bot cut off in {self.state}; back to {self._revert['state']}")
            undo = (self._snapshot(), self._delivering_line, self._revert)
            self._restore(self._revert)
            self._undo = undo
            self._just_reverted = True
            self._cancel("eval_filler", "eval_timeout")
        elif self.delivering and self.state in _PROMPT_STATES:
            self._undo = (self._snapshot(), self._delivering_line, None)
            self._go(State.LISTENING if self.state != State.CLARIFY else State.CLARIFY)
        self.delivering = False
        self._revert = None
        return self._out()

    def user_turn(self, text: str, silence_secs: float = 0.0) -> list[Action]:
        """A complete candidate turn. `silence_secs`: how long they've already
        been silent (VAD + turn detection), which counts toward the hold."""
        text = (text or "").strip()
        if not text:
            return []
        self._undo = None
        self._silence = silence_secs
        self._cancel(*LISTEN_TIMERS)
        s = self.state
        if s in (State.GREETING, State.READY_CHECK):
            self._on_ready_reply(text)
        elif s in (State.LISTENING, State.EXTENDED_WAIT, State.ANYTHING_TO_ADD, State.ASK_QUESTION, State.FOLLOW_UP):
            self._go(State.LISTENING)
            self._on_answer_piece(text)
        elif s == State.NEED_TIME:
            self._on_need_time_reply(text)
        elif s == State.CLARIFY:
            self._on_confirmation(text)
        elif s == State.EVALUATING:
            self._go(State.LISTENING)
            self._on_answer_piece(text)
        elif s in (State.CLOSING, State.REPORT, State.DONE):
            self._say(LINES["ended"])
        self._just_reverted = False
        return self._out()

    def empty_turn(self) -> list[Action]:
        """A turn ended with no words (all dropped as noise, echo or hallucination).
        If it had cut the bot off, the cut was a false alarm: say the line again.
        (Seen live: the old reply was re-used as the answer instead.)"""
        undo, self._undo = self._undo, None
        if undo and self.state not in (State.CLOSING, State.REPORT, State.DONE):
            snap, line, revert = undo
            self.false_cuts += 1
            logger.info("[ctrl] the cut-off had no words (noise or echo); saying the line again")
            self._restore(snap)
            self._revert = revert
            self._just_reverted = False
            if line:
                self._say(line)
        return self._out()

    def timer(self, name: str) -> list[Action]:
        s = self.state
        if name == "ready_reprompt" and s == State.READY_CHECK and not self.ready_reprompted:
            self.ready_reprompted = True
            self._say(LINES["ready_reprompt"])
        elif name == "no_speech":
            self._on_no_speech()
        elif name == "need_time_reply" and s == State.NEED_TIME:
            self._go(State.EXTENDED_WAIT)
            self._timer("extended_wait", self.cfg.extended_wait_secs)
        elif name == "extended_wait" and s == State.EXTENDED_WAIT:
            # If they start answering while we say "let's move on", go back.
            self._revert = self._snapshot()
            self._record("unanswered")
            self._next_question(opener=LINES["move_on"])
        elif name == "answer_hold" and s == State.LISTENING and self.answer_parts:
            self._evaluate()
        elif name == "max_answer" and self.question is not None:
            logger.info(f"[ctrl] q{self.q_index + 1}: answer passed {self.cfg.max_answer_secs}s")
            self.force_move_on = True
        elif name == "eval_filler" and s == State.EVALUATING and self.pending_turn_id:
            # Short spoken ack while the evaluator works: first audio comes
            # fast even though evaluation takes 2-3 s. Varied so it isn't robotic.
            # Only after a real answer, rarely, and never the same one twice
            # (the TL: "Don't keep saying got it thanks").
            words = len(self.pending_answer.split())
            if (
                not self.quiet
                and words >= self.cfg.filler_min_words
                and self.turn_seq - self._last_filler_turn >= self.cfg.filler_every
            ):
                fillers = [f for f in LINES["fillers"] if f != self._last_filler]
                f = fillers[self._filler_n % len(fillers)]
                self._filler_n += 1
                self._last_filler, self._last_filler_turn = f, self.turn_seq
                self._say(f)
                self._filler_said = True
        elif name == "eval_timeout" and s == State.EVALUATING and self.pending_turn_id:
            logger.warning(f"[ctrl] evaluator timed out for {self.pending_turn_id}; moving on")
            self._apply_result(
                EvalResult.default_next(self.pending_turn_id, self.pending_answer, "evaluator timeout")
            )
        elif name == "incomplete" and s == State.ANYTHING_TO_ADD and not self.delivering:
            self.awaiting = "anything_to_add"
            self.history.append(("bot", LINES["anything_to_add"]))
            self._say(LINES["anything_to_add"])
        return self._out()

    def eval_done(self, result: EvalResult) -> list[Action]:
        if result.turn_id != self.pending_turn_id:
            logger.info(f"[ctrl] discarding stale evaluation {result.turn_id}")
            return []
        self._cancel(*EVAL_TIMERS)
        self._apply_result(result)
        return self._out()

    # ------------------------------------------------------------- handlers

    def _greeting(self) -> str:
        p = self.plan
        name = f", {p.first_name}" if p.first_name else ""
        role = f" for the {p.role} role" if p.role else ""
        return (
            f"Hi{name}. I'm the AI interviewer{role}. "
            f"I'll ask you {len(p.questions)} questions, and you can take your time with each answer. "
            "Are you ready to begin?"
        )

    def _on_ready_reply(self, text: str):
        intent = intents.detect(text)
        if intent in ("identity", "salary", "process"):
            self._say(f"{self._fixed_answer(intent)} Are you ready to begin?")
        elif intent == "repeat":
            self._say(self._greeting())
        elif intents.is_negative(text) or intent in ("need_time", "doubt"):
            self._say(LINES["ready_wait"] if intent != "doubt" else LINES["doubt"])
        else:
            # Anything else ("yes", "sure", "let's go", even a stray sentence) starts it.
            self._revert = self._snapshot()
            self.started_at = self.started_at or self.clock()
            self._next_question(reply=text)

    def _fixed_answer(self, intent: str) -> str:
        if intent == "identity":
            return LINES["identity"]
        if intent == "salary":
            return LINES["salary"]
        if intent == "process":
            mins = round(self.cfg.time_limit_secs / 60)
            return f"There are {len(self.plan.questions)} questions, and the interview takes up to {mins} minutes."
        return LINES["cant_answer"]

    def _on_answer_piece(self, text: str):
        q = self.question
        if q is None:  # e.g. reconnected before the first question
            self._on_ready_reply(text)
            return
        if intents.mentions_outside_help(text):
            self._outside_help(text)
            return
        intent = intents.detect(text)
        if self.answer_parts and intent not in (
            "repeat", "previous", "need_time", "doubt", "go_on", "clarify_question",
            "hear_check", "read_back", "complaint", "end_request",
        ):
            intent = None  # mid-answer, short phrases are part of the answer

        if self.awaiting == "candidate_question":
            self.awaiting = None
            if intent == "repeat":
                self._repeat()
            elif intent in ("identity", "salary", "process"):
                self._say(f"{self._fixed_answer(intent)} {LINES['continue']}")
            else:
                # "I have a doubt" -> their question; usually about the question.
                self._clarify(text)
            return
        if self.awaiting == "repeat_confirm":
            self.awaiting = None
            if intents.is_affirmative(text) or intent == "repeat":
                self._repeat()
                return
            if intents.is_negative(text):
                self._say(LINES["go_ahead"])
                return
            # Otherwise they've started answering: fall through.
        if self.awaiting == "done_check":
            self.awaiting = None
            if (intents.is_affirmative(text) or _says_done(text)) and self.answer_parts:
                self._evaluate()
                return
            if intents.is_negative(text) and len(text.split()) <= 4:
                self._go(State.LISTENING)
                self._say(LINES["go_on"])
                return
            # Otherwise they simply carried on answering: fall through.
        if self.awaiting == "anything_to_add":
            if intents.is_negative(text) or _says_done(text):
                self.awaiting = None
                self._finish_question(self.held_result)
                return
            self.awaiting = None
        if self.awaiting == "skip_confirm":
            self.awaiting = None
            if intents.is_affirmative(text) or intent == "skip":
                self._flag("declined to answer", text[:120])
                self.history.append(("candidate", text))
                self._record("skipped")
                self._next_question(opener=LINES["skip"])
                return
            if intents.is_negative(text) and len(text.split()) <= 4:
                self._go(State.LISTENING)
                self._say(LINES["go_ahead"])
                return
        if self.awaiting == "end_confirm":
            self.awaiting = None
            if intents.is_affirmative(text) or intent == "end_request":
                self._flag("candidate ended the interview early", f"at question {self.q_index + 1}")
                self._record("unanswered")
                self._close(LINES["end_ok"])
                return
            self._go(State.ASK_QUESTION)
            self._say(f"{LINES['end_cancel']} {self._last_bot_line()}")
            return

        if not self.answer_parts and is_backchannel(text, self._backchannels):
            logger.info(f"[ctrl] filler {text!r}; still waiting for the answer")
            self._timer("no_speech", self.cfg.no_speech_prompt_secs)
            return

        if intent == "clarify_question":
            self._clarify(text)
        elif intent == "hear_check":
            if self.answer_parts:
                self._go(State.LISTENING)
                self._say(LINES["hear_go_on"])
            else:
                self._go(State.ASK_QUESTION)
                self._say(f"{LINES['hear_yes']} {self._last_bot_line()}")
        elif intent == "read_back":
            heard = " ".join(self.answer_parts) or next((t for w, t in reversed(self.history) if w == "candidate"), "")
            words = heard.split()
            line = f"You said: {' '.join(words[:35])}{'...' if len(words) > 35 else ''}." if heard else LINES["not_heard_yet"]
            self._go(State.LISTENING)
            self._say(f"{line} {LINES['continue']}")
        elif intent == "complaint":
            self.quiet = True
            logger.info(f"[ctrl] candidate complained: {text!r}; no more fillers")
            if self.answer_parts:
                self._go(State.LISTENING)
                self._say(f"{LINES['sorry']} {LINES['continue']}")
            else:
                self._go(State.ASK_QUESTION)
                self._say(f"{LINES['sorry']} {self._last_bot_line()}")
        elif intent == "refuse":
            self.awaiting = "skip_confirm"
            self._go(State.LISTENING)
            self._say(LINES["refuse"])
        elif intent == "end_request":
            self.awaiting = "end_confirm"
            self._go(State.LISTENING)
            self._say(LINES["end_confirm"])
        elif intent == "skip":
            self.history.append(("candidate", text))
            self._record("skipped")
            self._next_question(opener=LINES["skip"])
        elif intent == "go_on":
            self._go(State.LISTENING)
            self._say(LINES["go_on"])
        elif intent == "previous":
            self._revisit()
        elif intent == "repeat":
            self._repeat()
        elif intent == "need_time":
            self._go(State.EXTENDED_WAIT)
            self._say(LINES["take_time"])
        elif intent == "hint_request":
            self._say(LINES["hint"])
        elif intent in ("identity", "salary", "process"):
            self._say(f"{self._fixed_answer(intent)} {LINES['continue']}")
        elif intent == "doubt":
            self.awaiting = "candidate_question"
            self._say(LINES["doubt"])
        elif intent == "dont_know":
            # "I don't know... oh wait" during our reply -> back to this question.
            self._revert = self._snapshot()
            self.history.append(("candidate", text))
            self._record("dont_know")
            self._next_question(opener=LINES["dont_know"])
        else:
            self.answer_parts.append(text)
            words = len(" ".join(self.answer_parts).split())
            if (
                self._silence >= self.cfg.done_check_silence_secs
                and not self.done_checked
                and not self.force_move_on
            ):
                # The turn ended on Smart Turn's timeout: they may have trailed
                # off mid-thought. Ask instead of guessing.
                self.done_checked = True
                self.awaiting = "done_check"
                self._go(State.LISTENING)
                self._say(LINES["done_check"])
                return
            # Silence already spent waiting for Smart Turn counts toward the hold
            # (seen live: 2.7 s of turn detection + a full 1.2 s hold = 6 s reply).
            hold = self.cfg.answer_hold_secs - self._silence
            if words >= self.cfg.hold_min_words and hold > 0.05 and not self.force_move_on:
                self._timer("answer_hold", hold)
            else:
                self._evaluate()

    def _on_need_time_reply(self, text: str):
        intent = intents.detect(text)
        if intents.is_affirmative(text) or intent == "need_time":
            self._go(State.EXTENDED_WAIT)
            self._say(LINES["take_time"])
        elif intents.is_negative(text):
            self._go(State.EXTENDED_WAIT)
            self._say(LINES["go_ahead"])
        else:
            self._go(State.LISTENING)
            self._on_answer_piece(text)

    def _on_confirmation(self, text: str):
        """Reply to "Just to confirm, did you say X?"."""
        held, term, heard = self.held_result, self.clarify_term, self.clarify_heard
        if intents.is_affirmative(text) and not intents.is_negative(text) and held is not None:
            # Confirmed: apply the word and carry on with the held decision; no
            # second LLM call (seen live: the re-check took 21 s and timed out).
            self.held_result = None
            if heard and term:
                held.corrections.append(Correction(**{"from": heard, "to": term}))
                held.interpreted_answer = held.interpreted_answer.replace(heard, term)
                self.history = [
                    (w, t.replace(heard, term) if w == "candidate" else t) for w, t in self.history
                ]
                self.pending_answer = self.pending_answer.replace(heard, term)
            held.unclear_terms = []
            self._go(State.EVALUATING)
            self._decide(held)
            return
        if not intents.is_negative(text) and is_backchannel(text, self._backchannels):
            return  # "thank you", "hmm": not a reply to the question
        self.held_result = None
        # Not confirmed: re-score with their own words, and say so plainly.
        self.history = [h for h in self.history if h[1] != self.pending_answer or h[0] != "candidate"]
        self._evaluate(
            answer=f"{self.pending_answer}\n[Asked if they said '{term}', the candidate replied: {text}. "
            f"Use their words, not '{term}'.]"
        )

    def _on_no_speech(self):
        s = self.state
        if s == State.CLARIFY:
            # No reply to the confirmation: accept the best guess, unverified.
            held = self.held_result
            self.held_result = None
            if held is not None:
                held.flags.append(f"unverified: {self.clarify_term}")
            self._finish_question(held)
        elif s == State.LISTENING and self.awaiting == "done_check":
            # No reply to "Is that your complete answer?": take it as yes.
            self.awaiting = None
            if self.answer_parts:
                self._evaluate()
        elif s == State.LISTENING and self.awaiting == "anything_to_add":
            self.awaiting = None
            self._finish_question(self.held_result)
        elif s == State.LISTENING and not self.answer_parts and not self.delivering:
            if self.need_time_used:
                self._go(State.EXTENDED_WAIT)
                self._timer("extended_wait", self.cfg.extended_wait_secs)
            else:
                self.need_time_used = True
                self._go(State.NEED_TIME)
                self._say(LINES["need_time"])

    def _repeat(self):
        """1st request: verbatim. 2nd: reworded (capped per interview).
        Later: "give it your best attempt" + a flag for the recruiter."""
        # A half-started answer before "can you repeat?" is abandoned, not
        # part of the real answer (seen live: "Can you..." stuck to the answer).
        self.answer_parts = []
        self.repeat_requests_q += 1
        self.repeat_requests_total += 1
        n = self.repeat_requests_q
        line = self._last_bot_line()
        self._go(State.ASK_QUESTION)
        if n >= self.cfg.repeats_before_best_attempt:
            self._flag("repeated clarification requests", f"{n} on this question")
            self._go(State.LISTENING)
            self._say(LINES["best_attempt"])
        elif n == 2 and self.rephrases_total < self.cfg.max_rephrases and self.question is not None:
            self.rephrases_total += 1
            q = self.question
            rubric = q.expected if line == q.q else []
            self._acts.append(Rephrase(line, rubric, self.q_index + 1))
            self.delivering = True
            self._delivering_line = line
        else:
            if n == 2:
                logger.info("[ctrl] rephrase limit for the interview reached; repeating verbatim")
            self._say(f"Sure. {line}")
        if self.repeat_requests_total == self.cfg.flag_repeats_total:
            self._flag("frequent clarification requests", f"{self.repeat_requests_total} in the interview")

    def _outside_help(self, text: str):
        """'Claude told me the right answer': flag it, and say so, like a real interviewer."""
        self.outside_help_q += 1
        self._flag("candidate said they used outside help or AI", text[:120])
        self.answer_parts, self.answer_started = [], False
        if self.outside_help_q >= 2:
            self.history.append(("candidate", text))
            self._record("unanswered")
            self._next_question(opener=f"{LINES['own_words']} Let's move on.")
            return
        self._go(State.ASK_QUESTION)
        self._say(f"{LINES['own_words']} Let me ask again. {self._last_bot_line()}")

    def _clarify(self, asked: str):
        """'Which project are you asking about?' -> a scoped, hint-free reply.
        Shares the repeat budget, so it can't be used to fish for hints."""
        self.repeat_requests_q += 1
        self.repeat_requests_total += 1
        line = self._last_bot_line()
        if self.repeat_requests_q >= self.cfg.repeats_before_best_attempt + 1:
            self._flag("repeated clarification requests", f"{self.repeat_requests_q} on this question")
            self._go(State.LISTENING)
            self._say(LINES["best_attempt"])
            return
        q = self.question
        # Knowledge questions: the rubric is the answer, so the reply must not use it.
        rubric = q.expected if q is not None and q.topic in ("jd_skill", "fundamentals") else []
        answers = [t for w, t in self.history if w == "candidate"] + list(self.answer_parts)
        self._go(State.ASK_QUESTION)
        self._acts.append(
            Clarify(line, asked, rubric, q.context if q else "", answers, self._earlier())
        )
        self.delivering = True
        self._delivering_line = line
        if self.repeat_requests_total == self.cfg.flag_repeats_total:
            self._flag("frequent clarification requests", f"{self.repeat_requests_total} in the interview")

    def _revisit(self):
        """'Can you repeat the previous question?' -> declined: once the
        interview has moved on, it doesn't go back."""
        if self._just_reverted or self.q_index <= 0:
            # We were cut off starting the next question, so the "previous"
            # question to them is the one we're still on.
            self._repeat()
            return
        self.go_back_requests += 1
        logger.info(f"[ctrl] asked to go back from q{self.q_index + 1}; declined")
        if self.go_back_requests >= self.cfg.flag_go_back_requests:
            self._flag("asked to go back to earlier questions", f"{self.go_back_requests} times")
        self._go(State.ASK_QUESTION)
        self._say(f"{LINES['no_going_back']} {self._last_bot_line()}")

    # ------------------------------------------------------------ evaluation

    def _evaluate(self, answer: str | None = None):
        q = self.question
        if q is None:
            return
        self._revert = self._snapshot()
        self.turn_seq += 1
        self._filler_said = False
        turn_id = f"q{q.id}-t{self.turn_seq}"
        answer = answer if answer is not None else " ".join(self.answer_parts)
        self.pending_turn_id = turn_id
        self.pending_answer = answer
        req = EvalRequest(
            turn_id=turn_id,
            question=q,
            history=list(self.history) + [("candidate", answer)],
            answer=answer,
            vocab=self.plan.stt_vocab(),
            followups_used=self.followups,
            earlier=self._earlier(),
            level=self.plan.level,
        )
        self._go(State.EVALUATING)
        self._acts.append(Evaluate(req))
        self._timer("eval_filler", self.cfg.evaluator_filler_secs)
        self._timer("eval_timeout", self.cfg.evaluator_timeout_secs)

    def _apply_result(self, r: EvalResult):
        # Advisory flags for the recruiter; they never change the flow by themselves.
        for f in r.flags:
            self._flag(f"evaluator: {f}", r.notes)
        if r.verdict == "suspicious":
            self._flag("suspicious answer (advisory)", r.notes)
        self.pending_turn_id = None
        self.history.append(("candidate", self.pending_answer))
        self.answer_parts = []
        q = self.question
        assert q is not None

        if self.force_move_on:
            self._finish_question(r, opener=LINES["max_answer"])
            return
        # Intents the rules missed but the evaluator caught (Phase 4).
        if r.intent == "repeat":
            self._repeat()
            return
        if r.intent == "dont_know":
            self._record("dont_know", r)
            self._next_question(opener=LINES["dont_know"])
            return
        if r.intent == "need_time":
            self._go(State.EXTENDED_WAIT)
            self._say(LINES["take_time"])
            return
        if r.intent == "hint_request":
            self._go(State.LISTENING)
            self._say(LINES["hint"])
            return
        if r.intent == "injection_attempt":
            self._go(State.ASK_QUESTION)
            self._say(f"{LINES['stay_on_topic']} {self._last_bot_line()}")
            return
        if r.intent in ("other_question", "off_topic"):
            # Anything that isn't an answer ("your voice is breaking", "I'm
            # nervous", "someone is calling me"): one natural reply from the
            # LLM instead of a fixed line per case, then back to the question.
            if r.intent == "off_topic":
                self.off_topic_total += 1
                if self.off_topic_total == 2:
                    self._flag("other speech or reading aloud", "off-topic replies twice")
            if self.history and self.history[-1] == ("candidate", self.pending_answer):
                self.history.pop()
            q = self.question
            rubric = q.expected if q is not None and q.topic in ("jd_skill", "fundamentals") else []
            line = self._last_bot_line()
            self._go(State.ASK_QUESTION)
            self._acts.append(Respond(self.pending_answer, line, rubric))
            self.delivering = True
            self._delivering_line = line
            return
        if r.intent in ("clarify_question", "earlier_question"):
            # Not an answer to this question: take it back out of the record.
            if self.history and self.history[-1] == ("candidate", self.pending_answer):
                self.history.pop()
            if r.intent == "clarify_question":
                self._clarify(self.pending_answer)
            else:
                self._flag("answered an earlier question", self.pending_answer[:120])
                self._revisit()
            return
        self._decide(r)

    def _decide(self, r: EvalResult):
        """What to do with a real answer: confirm a word, wait, follow up, or move on."""
        q = self.question
        assert q is not None
        high = [t for t in r.unclear_terms if t.importance == "high"]
        if (
            high
            and r.verdict != "unanswered"  # no "did you say X?" about noise
            and self.confirmations_q == 0
            and self.confirmations_total < self.cfg.max_confirmations
        ):
            self.confirmations_q += 1
            self.confirmations_total += 1
            self.clarify_term = high[0].best_guess
            self.clarify_heard = high[0].heard
            self.held_result = r
            line = f"Just to confirm, did you say {high[0].best_guess}?"
            self.history.append(("bot", line))
            self._go(State.CLARIFY)
            self._say(line)
            return
        if not r.complete:
            self.held_result = r
            self._go(State.ANYTHING_TO_ADD)
            self._timer("incomplete", self.cfg.incomplete_prompt_secs)
            return
        if (
            r.action == "verify"
            and r.follow_up_question
            and not self.verified_q
            and self.verifications_total < self.cfg.max_verifications
        ):
            # Re-checking a claim is worth asking even when follow-ups are used up.
            self.verified_q = True
            self.verifications_total += 1
            self._flag("re-verification asked", r.follow_up_question)
            self.history.append(("bot", r.follow_up_question))
            self.answer_started = False
            self._go(State.FOLLOW_UP)
            self._say(r.follow_up_question)
            return
        if (
            r.action in ("follow_up", "verify")
            and r.follow_up_question
            and self.followups < q.max_followups
        ):
            self.followups += 1
            self.history.append(("bot", r.follow_up_question))
            self.answer_started = False
            self._go(State.FOLLOW_UP)
            self._say(r.follow_up_question)
            return
        self._finish_question(r)

    def _finish_question(self, r: EvalResult | None, opener: str | None = None):
        self._record("answered", r)
        last_reply = next((t for w, t in reversed(self.history) if w == "candidate"), "")
        self._next_question(reply=last_reply, opener=opener)

    # --------------------------------------------------------------- flow

    def _next_question(self, reply: str = "", opener: str | None = None):
        self._cancel("max_answer")
        previous = self.question
        if self.q_index + 1 >= len(self.plan.questions) or self._time_up():
            self._close(opener)
            return
        self.q_index += 1
        q = self.question
        self.followups = self.confirmations_q = 0
        self.history = [("bot", q.q)]
        self.answer_parts, self.answer_started = [], False
        self.need_time_used = self.force_move_on = False
        self.awaiting, self.held_result, self.clarify_term, self.clarify_heard = None, None, None, None
        self.repeat_requests_q = 0
        self.done_checked = False
        self.verified_q = False
        self.outside_help_q = 0
        self._go(State.ASK_QUESTION)
        if opener:
            # Fixed opener + the question verbatim: no LLM needed.
            self._say(f"{opener} {q.q}")
        elif self._filler_said:
            # "Okay, got it." was already said: the question itself, no LLM
            # opener (avoids a double ack and saves ~1 s).
            self._filler_said = False
            self._say(q.q)
        else:
            self._acts.append(Ask(q, self.q_index + 1, reply, previous))
            self.delivering = True
            self._delivering_line = "Thanks for that. " + q.q  # ~ what will be said

    def _time_up(self) -> bool:
        return self.started_at is not None and self.clock() - self.started_at >= self.cfg.time_limit_secs

    def _close(self, opener: str | None = None):
        p = self.plan
        name = f", {p.first_name}" if p.first_name else ""
        why = "That's all the questions I have." if not self._time_up() else "We've reached the end of our time."
        line = (
            f"{(opener + ' ') if opener else ''}Thank you{name}. {why} "
            "A recruiter will review this interview and get back to you. Goodbye."
        )
        self._go(State.CLOSING)
        self._say(line)

    # ------------------------------------------------------------ persistence

    _SAVED = (
        "q_index", "followups", "confirmations_q", "confirmations_total", "history", "answer_parts",
        "started_at", "need_time_used", "ready_reprompted", "turn_seq", "results", "flags",
        "repeat_requests_q", "repeat_requests_total", "rephrases_total", "go_back_requests",
        "past", "verifications_total", "verified_q", "quiet", "off_topic_total", "outside_help_q",
    )

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self._SAVED}
        d["state"] = str(self.state)
        d["candidate"] = self.plan.candidate
        d["saved_at"] = self.clock()
        return d

    def load_dict(self, d: dict):
        for k in self._SAVED:
            if k in d:
                setattr(self, k, d[k])
        self.history = [tuple(x) for x in self.history]
        self.past = [tuple(x) for x in self.past]
        self.state = State(d.get("state", State.GREETING))


def _says_done(text: str) -> bool:
    t = f" {intents.normalize(text)} "
    return any(p in t for p in (" that's it ", " that's all ", " nothing ", " i'm done ", " that is all "))
