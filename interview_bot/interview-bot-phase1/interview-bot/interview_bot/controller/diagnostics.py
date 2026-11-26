"""Diagnostics: logs how loud the incoming mic audio is, every few seconds.

Answers "is my audio reaching the server, and is it loud enough for the VAD?"
without guessing. Set AUDIO_LEVEL_LOG_SECS=0 to turn it off.
"""

from __future__ import annotations

import math
import time

import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _dbfs(x: float) -> float:
    return 20 * math.log10(x) if x > 0 else -120.0


class AudioLevelLogger(FrameProcessor):
    def __init__(self, every_secs: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self._every = every_secs
        self._reset()

    def _reset(self):
        self._t0 = time.monotonic()
        self._frames = 0
        self._peak = 0.0
        self._sumsq = 0.0
        self._n = 0
        self._sr = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._every > 0 and isinstance(frame, InputAudioRawFrame):
            a = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            if a.size:
                self._frames += 1
                self._sr = frame.sample_rate
                self._peak = max(self._peak, float(np.abs(a).max()))
                self._sumsq += float((a * a).sum())
                self._n += a.size
            if time.monotonic() - self._t0 >= self._every:
                rms = math.sqrt(self._sumsq / self._n) if self._n else 0.0
                logger.info(
                    f"[mic-in] frames={self._frames} sr={self._sr} "
                    f"peak={_dbfs(self._peak):.0f} dBFS rms={_dbfs(rms):.0f} dBFS"
                )
                self._reset()
        await self.push_frame(frame, direction)
