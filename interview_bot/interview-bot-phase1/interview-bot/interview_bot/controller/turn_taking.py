"""Barge-in policy: when does candidate speech count as a new turn (and cut TTS)?

Plugs into Pipecat's user-turn system as a *start* strategy. End-of-turn is
decided separately by Smart Turn v3 (TurnAnalyzerUserTurnStopStrategy).

Rules (thresholds from config.py):
  * Bot silent -> any VAD-confirmed speech starts a turn at once.
  * Bot speaking, candidate started within EARLY_BARGE_IN_WINDOW_SECS of the bot
    starting -> cut TTS at once (the bot jumped in before the answer was done).
  * Candidate still talking when the bot starts -> cut TTS.
  * Bot speaking otherwise ("overlap"), judged by length *and* words:
      - speech < BARGE_IN_MIN_SECS        -> backchannel: ignore, drop its transcript
      - speech > BARGE_IN_HARD_SECS       -> real interruption: cut TTS at once
      - in between                        -> wait for the transcript: only
        backchannel words ("okay", "hmm", "yeah", "haan", ...) -> ignore;
        anything else ("sorry, wait", "actually...") -> cut TTS.
    Length alone can't work: a natural "okay" measured 0.7 s in testing, longer
    than the 0.5 s floor, while the start of a real interruption is just as short.

Requiring sustained speech, or real words, while TTS plays is also the echo
guard: residual speaker echo that survives browser AEC is short and choppy.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_start import BaseUserTurnStartStrategy

from interview_bot.services.stt import normalize

import re  # noqa: E402

# Short words that are never just "listening noises" when said over the bot.
_YES_NO = re.compile(r"^\W*(yes|yeah|yep|yup|yes please|no|nope|sure|okay|ok)\W*$", re.I)
_COMMAND = re.compile(r"\b(no|wait|stop|sorry|hello|excuse|pardon|repeat|hold on)\b", re.I)

CONTINUATION_SECS = 3.0  # bot speech this soon after it stopped is the same reply going on

DEFAULT_BACKCHANNELS = (
    "okay|ok|okay okay|yeah|yes|yep|yup|hmm|hm|mm|mhm|mm hmm|mmhmm|uh huh|right|"
    "sure|alright|all right|got it|i see|nice|cool|great|haan|han|ha|achha|acha|"
    "accha|theek hai|thik hai|ji|haan ji|bye|ok bye|okay bye|thanks|thank you|thank you so much|"
    "perfect|awesome|wow|nice one"
)


def is_backchannel(text: str, phrases: set[str]) -> bool:
    """True if the whole transcript is just listening noises, e.g. "Okay." or "Hmm, yeah"."""
    norm = normalize(text).replace("-", " ")
    if not norm:
        return True
    if norm in phrases:
        return True
    # "okay yeah", "hmm right": every word is itself a backchannel.
    words = norm.split()
    return len(words) <= 3 and all(w in phrases for w in words)


def parse_backchannels(spec: str) -> set[str]:
    return {normalize(p).replace("-", " ") for p in spec.split("|") if p.strip()}


class BargeInStartStrategy(BaseUserTurnStartStrategy):
    def __init__(
        self,
        *,
        barge_in_min_secs: float = 0.5,
        barge_in_hard_secs: float = 1.0,
        early_window_secs: float = 0.3,
        vad_stop_secs: float = 0.0,
        transcript_wait_secs: float = 1.5,
        backchannels: set[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        echo_guard=None,
        expects_answer: Callable[[], bool] | None = None,
        reply_pending: Callable[[], bool] | None = None,
        noisy: Callable[[], bool] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._expects_answer = expects_answer or (lambda: False)
        self._reply_pending = reply_pending or (lambda: False)  # a reply is being prepared
        self._noisy = noisy or (lambda: False)  # STT keeps throwing sound away as noise
        self._noise_noted = False
        self._echo = echo_guard  # strict mode after repeated false cut-offs
        self._min_secs = barge_in_min_secs
        self._hard_secs = max(barge_in_hard_secs, barge_in_min_secs)
        self._early_window = early_window_secs
        # VAD reports "stopped" only after stop_secs of silence, so its
        # "speaking" state outlasts the real speech by that much.
        self._vad_stop_secs = vad_stop_secs
        self._transcript_wait = transcript_wait_secs
        self._backchannels = backchannels or parse_backchannels(DEFAULT_BACKCHANNELS)
        self._clock = clock

        self._bot_speaking = False
        self._bot_started_at = 0.0
        self._bot_stopped_at = -99.0
        self._continuation = False  # the bot resumed its own reply (lead-in -> question)
        self._user_vad_active = False
        self._turn_active = False

        # One "overlap" = candidate speech that started while the bot was talking.
        self._overlap_t0: float | None = None  # when VAD reported its start
        self._overlap_start_secs = 0.0  # speech VAD heard before reporting
        self._hard_timer: asyncio.Task | None = None
        self._drop_next_transcript = False  # too short: backchannel by length
        self._await_words = False  # mid-length: let the words decide
        self._words_timer: asyncio.Task | None = None

    # -- turn lifecycle callbacks from Pipecat --------------------------------

    async def handle_user_turn_started(self):
        await super().handle_user_turn_started()
        self._turn_active = True
        await self._clear_overlap()

    async def handle_user_turn_stopped(self):
        await super().handle_user_turn_stopped()
        self._turn_active = False

    async def cleanup(self):
        await self._clear_overlap()
        await super().cleanup()

    # -- frame handling ---------------------------------------------------------

    async def process_frame(self, frame: Frame):
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._bot_started_at = self._clock()
            # Live: the question started 2.4 s after its lead-in ended, the early window
            # reopened and the bot's own echo ("No") cut it off three times.
            self._continuation = self._bot_started_at - self._bot_stopped_at < CONTINUATION_SECS
            if self._user_vad_active and not self._turn_active and not self._strict():
                logger.debug("barge-in: candidate still talking as bot started -> cut TTS")
                await self._start_turn()
            elif self._user_vad_active and self._turn_active and not self._continuation and not self._strict():
                # Live: their turn was still going (they had paused, then went on) when a reply
                # queued earlier started playing, and the bot talked over them for 4 s.
                logger.debug("barge-in: reply started during the candidate's turn -> stop it")
                await self.broadcast_frame(InterruptionFrame)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_at = self._clock()
            if self._hard_timer and self._user_vad_active and not self._turn_active:
                # Bot finished while the candidate was mid-speech: it's their turn now.
                await self._start_turn()

        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_vad_active = True
            if self._turn_active:
                return
            if not self._bot_speaking and self._guard_reply():
                # Noisy room and a reply on its way: sound alone doesn't cancel it - real
                # words (or 2 s of steady speech) do. Live (Nov 18): background voices
                # cancelled every reply and the candidate heard nothing for 90 s.
                logger.debug("barge-in: sound while a reply is coming (noisy room) -> checking the words")
                await self._clear_overlap()
                self._overlap_t0 = self._clock()
                self._overlap_start_secs = frame.start_secs
                delay = max(0.0, max(self._hard_secs, 2.0) - frame.start_secs) + self._vad_stop_secs
                self._hard_timer = self.create_task(self._hard_cut_after(delay), "barge_in_hard")
            elif not self._bot_speaking:
                await self._start_turn()
            elif (self._clock() - self._bot_started_at <= self._early_window and not self._strict()
                  and not self._continuation):
                logger.debug("barge-in: early (bot just started) -> cut TTS")
                await self._start_turn()
            else:
                await self._clear_overlap()
                self._overlap_t0 = self._clock()
                self._overlap_start_secs = frame.start_secs
                # Still speaking `hard_secs` after the speech began -> interruption.
                # Echo mode (after false cut-offs): still cut on sustained speech, just later.
                # (Words-only waited for the candidate to pause, so the bot talked over them.)
                hard = max(self._hard_secs, 2.0) if self._strict() else self._hard_secs
                delay = max(0.0, hard - frame.start_secs) + self._vad_stop_secs
                self._hard_timer = self.create_task(self._hard_cut_after(delay), "barge_in_hard")

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_vad_active = False
            if self._overlap_t0 is None or self._turn_active:
                return
            await self._cancel(self._hard_timer)
            self._hard_timer = None
            spoken = (
                self._clock() - self._overlap_t0 + self._overlap_start_secs - self._vad_stop_secs
            )
            self._overlap_t0 = None
            if spoken < self._min_secs:
                logger.debug(f"barge-in: {spoken:.2f}s burst while bot speaking -> backchannel")
                self._drop_next_transcript = True
            else:
                logger.debug(f"barge-in: {spoken:.2f}s while bot speaking -> checking the words")
                self._await_words = True
                self._words_timer = self.create_task(self._words_timeout(), "barge_in_words")

        elif isinstance(frame, TranscriptionFrame):
            if self._turn_active:
                return
            if (self._drop_next_transcript or self._await_words) and _YES_NO.match(frame.text or "") \
                    and self._expects_answer():
                # The bot asked a yes/no question ("shall we skip question 3?"): a short "Yes"
                # over it is the answer (live: two of them were dropped as backchannels).
                self._drop_next_transcript = self._await_words = False
                await self._cancel(self._words_timer)
                self._words_timer = None
                logger.debug(f"barge-in: {frame.text!r} answers the bot's yes/no question -> their turn")
                await self._start_turn()
            elif self._drop_next_transcript and _COMMAND.search(frame.text or ""):
                # Live: "Wait!" (0.5 s) over the goodbye was ignored as a backchannel.
                self._drop_next_transcript = False
                logger.debug(f"barge-in: short command {frame.text!r} -> cut TTS")
                await self._start_turn()
            elif self._drop_next_transcript:
                self._drop_next_transcript = False
                logger.info(f"Ignored backchannel transcript: {frame.text!r}")
                await self.trigger_reset_aggregation()
            elif self._await_words:
                self._await_words = False
                await self._cancel(self._words_timer)
                self._words_timer = None
                if is_backchannel(frame.text, self._backchannels):
                    logger.info(f"Ignored backchannel words: {frame.text!r}")
                    await self.trigger_reset_aggregation()
                else:
                    logger.debug(f"barge-in: real words {frame.text!r} -> cut TTS")
                    await self._start_turn()
            elif not self._bot_speaking:
                # VAD missed the start but STT heard words: still a turn.
                await self._start_turn()

    # -- helpers ------------------------------------------------------------------

    def _guard_reply(self) -> bool:
        if not self._reply_pending():
            return False
        # Once noticed, the room stays "noisy" for this interview (background noise rarely stops).
        noisy = self._noise_noted or self._strict() or self._noisy()
        if noisy and not self._noise_noted:
            self._noise_noted = True
            logger.info("[noise] background voices or noise: only real words stop a reply now "
                        "(a quiet room or headphones help)")
        return noisy

    def _strict(self) -> bool:
        return bool(self._echo is not None and self._echo.require_words)

    async def _hard_cut_after(self, delay: float):
        await asyncio.sleep(delay)
        self._hard_timer = None
        if self._user_vad_active and not self._turn_active:
            logger.debug(f"barge-in: sustained speech > {self._hard_secs}s -> cut TTS")
            await self._start_turn()

    async def _words_timeout(self):
        # No transcript (e.g. the STT dropped it as noise): treat as a backchannel.
        await asyncio.sleep(self._transcript_wait)
        self._words_timer = None
        self._await_words = False

    async def _start_turn(self):
        await self._clear_overlap()
        await self.report_user_turn_started()

    async def _clear_overlap(self):
        self._overlap_t0 = None
        self._drop_next_transcript = False
        self._await_words = False
        hard, self._hard_timer = self._hard_timer, None
        words, self._words_timer = self._words_timer, None
        await self._cancel(hard)
        await self._cancel(words)

    async def _cancel(self, task: asyncio.Task | None):
        if task and not task.done() and task is not asyncio.current_task():
            await self.cancel_task(task)


# -- fast end of turn for short closed replies -------------------------------------

from pipecat.turns.user_stop import BaseUserTurnStopStrategy  # noqa: E402

CLOSED_REPLY_WORDS = {
    "yes", "yeah", "yep", "yup", "no", "nope", "nah", "okay", "ok", "sure", "haan", "ha", "nahi", "ji",
    "correct", "right", "fine", "alright", "ready", "go", "ahead", "please", "continue", "stop", "done",
}


class ShortReplyStopStrategy(BaseUserTurnStopStrategy):
    """Ends the turn as soon as the candidate stops on a short closed reply
    ("Yes", "No", "Okay, go ahead"). Seen live: Smart Turn marked a bare "Yes"
    as incomplete and the turn waited ~3 s before the bot answered."""

    def __init__(self, max_words: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._max_words = max_words
        self._text = ""
        self._vad = False

    async def handle_user_turn_started(self):
        self._text = ""

    async def handle_user_turn_stopped(self):
        self._text = ""

    async def process_frame(self, frame):
        await super().process_frame(frame)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad = False
            await self._check()
        elif isinstance(frame, TranscriptionFrame):
            self._text = f"{self._text} {frame.text}"
            await self._check()

    async def _check(self):
        if self._vad or not self._text.strip():
            return
        words = normalize(self._text).replace("-", " ").split()
        if 0 < len(words) <= self._max_words and all(w in CLOSED_REPLY_WORDS for w in words):
            logger.debug(f"turn end: short reply {self._text.strip()!r}")
            self._text = ""
            await self.report_user_turn_stopped()
