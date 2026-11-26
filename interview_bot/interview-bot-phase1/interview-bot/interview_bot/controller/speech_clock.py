"""Tracks whether the candidate is speaking right now, and for how long they've been silent.

VAD runs inside the user aggregator and sends its start/stop frames upstream
(to the STT). A tap placed just before the aggregator sees them on the way.
The session uses the clock so that "5 s of silence" means 5 s of real
silence, not 5 s since a timer happened to start. (Seen live: the
"Would you like a moment?" timer started while the candidate was already
talking, then fired during a pause mid-answer.)
"""

from __future__ import annotations

import time

from pipecat.frames.frames import Frame, TTSTextFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class SpeechClock:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self.user_speaking = False
        self.last_speech_end = clock()

    def started(self):
        self.user_speaking = True

    def stopped(self, stop_secs: float = 0.0):
        self.user_speaking = False
        # VAD reports the stop stop_secs after the speech actually ended.
        self.last_speech_end = self._clock() - stop_secs

    def silence_secs(self) -> float:
        return 0.0 if self.user_speaking else max(0.0, self._clock() - self.last_speech_end)


class SpeechClockTap(FrameProcessor):
    def __init__(self, clock: SpeechClock, **kwargs):
        super().__init__(**kwargs)
        self.clock = clock

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self.clock.started()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.clock.stopped(frame.stop_secs)
        await self.push_frame(frame, direction)


class PlayedTextTap(FrameProcessor):
    """After transport.output(): a sentence's TTSTextFrame comes out only once its audio
    has been played, so this tells the controller which sentences the candidate has heard.
    Live (Nov 10): "Okay." during the greeting, before "Are you ready to begin?" had been
    said, was taken as the answer to it and the interview started."""

    def __init__(self, on_played, **kwargs):
        super().__init__(**kwargs)
        self._on_played = on_played

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSTextFrame) and direction == FrameDirection.DOWNSTREAM and frame.text:
            try:
                self._on_played(frame.text)
            except Exception:  # noqa: BLE001 - never break the pipeline over bookkeeping
                pass
        await self.push_frame(frame, direction)
