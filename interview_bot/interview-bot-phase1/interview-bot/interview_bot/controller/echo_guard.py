"""Stops the bot from hearing (and answering) its own voice.

On laptop speakers the browser's echo cancellation leaves some of the bot's
audio in the mic. Seen live: the "candidate" said "Okay, got it." right after
the bot did, and echo cut off almost every question, which looped.

Two defences, shared between the pipeline pieces:
- EchoFilter drops a transcript that repeats what the bot just said while it
  was speaking (or a moment after).
- After repeated false cut-offs (speech with no real words), barge-in stops
  cutting on loudness alone and waits for real words.
"""

from __future__ import annotations

import time
from collections import deque

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from interview_bot.services.stt import normalize


class EchoGuard:
    def __init__(self, window_secs: float = 20.0, tail_secs: float = 2.5, false_cuts_limit: int = 2, clock=time.monotonic):
        self._lines: deque[tuple[float, str]] = deque(maxlen=12)
        self._window = window_secs
        self._tail = tail_secs
        self._limit = false_cuts_limit
        self._clock = clock
        self.bot_speaking = False
        self.bot_stopped_at = -1e9
        self.false_cuts = 0
        self.require_words = False

    def bot_said(self, text: str):
        if text.strip():
            self._lines.append((self._clock(), text))

    def false_cut(self):
        self.false_cuts += 1
        if self.false_cuts >= self._limit and not self.require_words:
            self.require_words = True
            logger.warning(
                f"[echo] {self.false_cuts} false cut-offs: from now on only real words interrupt the bot "
                "(echo from speakers? headphones help)"
            )

    def is_echo(self, text: str) -> bool:
        now = self._clock()
        if not (self.bot_speaking or now - self.bot_stopped_at <= self._tail):
            return False
        words = normalize(text).replace("-", " ").split()
        if not words or len(words) > 20:
            return False
        if not self.bot_speaking and len(words) < 3:
            # After the bot stops, a one- or two-word reply is the candidate answering
            # (seen live: "Yes" to "Just say yes or no." was dropped as echo).
            return False
        recent_text = normalize(" ".join(t for at, t in self._lines if now - at <= self._window)).replace("-", " ")
        if len(words) <= 8:
            # Short phrases must appear word for word in what the bot said: live,
            # "No, no" over "Okay, no problem. Thank you..." and "I don't want to answer"
            # over "...if you don't want to share, you can answer in general terms" were dropped.
            return f" {' '.join(words)} " in f" {recent_text} "
        recent = set(recent_text.split())
        return sum(w in recent for w in words) / len(words) >= 0.8


class EchoFilter(FrameProcessor):
    """Before the user aggregator: drops transcripts that are the bot's own voice."""

    def __init__(self, guard: EchoGuard, **kwargs):
        super().__init__(**kwargs)
        self._guard = guard

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._guard.bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._guard.bot_speaking = False
            self._guard.bot_stopped_at = self._guard._clock()
        elif isinstance(frame, TranscriptionFrame) and self._guard.is_echo(frame.text):
            logger.info(f"[echo] dropped the bot's own voice: {frame.text!r}")
            return
        await self.push_frame(frame, direction)
