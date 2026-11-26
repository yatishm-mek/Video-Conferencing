"""Cloned-voice TTS for the bot — Qwen3-TTS via the TTS microservice.

The 4 GB Qwen3-TTS base model lives in its own `qwen3tts` venv/process
(integration/tts_service). This Pipecat service is a thin client: for each text
chunk it POSTs to that service, gets raw PCM16 @ 24 kHz back, resamples to the
pipeline's output rate, and yields it as `TTSAudioRawFrame`s (200 ms pieces so it
stays interruptible).

Note: cloned voices are slow on the current GPU (~3–5 s per 1 s of audio), so the
first audio of each turn lags. The pipeline's interruption handling is unchanged.
"""

from __future__ import annotations

from typing import AsyncGenerator

import httpx
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService


class QwenCloneTTSService(TTSService):
    """Streams cloned-voice audio from the Qwen3-TTS microservice."""

    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:8099",
        voice: str = "rithika",
        style: str = "balanced",
        request_timeout: float = 60.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._url = url.rstrip("/")
        self._voice = voice
        self._style = style
        self._timeout = request_timeout

    async def _fetch(self, text: str) -> tuple[bytes, int]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(
                f"{self._url}/tts",
                json={"voice": self._voice, "text": text, "style": self._style},
            )
            r.raise_for_status()
            rate = int(r.headers.get("X-Sample-Rate", "24000"))
            return r.content, rate

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        yield None  # async-generator contract before awaiting
        await self.start_ttfb_metrics()
        await self.start_tts_usage_metrics(text)
        try:
            pcm, rate = await self._fetch(text)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[qwen-tts] synthesis failed ({e}); is the TTS microservice running at {self._url}?")
            yield ErrorFrame(f"Qwen TTS error: {e}", fatal=False)
            return
        if not pcm:
            return
        data = await self._resampler.resample(pcm, rate, self.sample_rate)
        await self.stop_ttfb_metrics()
        step = int(self.sample_rate * 0.2) * 2  # 200 ms pieces, still interruptible
        for i in range(0, len(data), step):
            yield TTSAudioRawFrame(
                audio=data[i : i + step], sample_rate=self.sample_rate, num_channels=1, context_id=context_id
            )
