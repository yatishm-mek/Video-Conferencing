"""Moonshine STT for the bot's own listening, via the Vosk-WS server.

The fine-tuned Moonshine model already runs as a Vosk-protocol WebSocket server
(custom-asr/asr_ws_server.py) so Jitsi can transcribe with it. Rather than load
the model a second time in the bot's process (different venv, transformers build),
this adapter reuses that running server: for each VAD-bounded utterance Pipecat
hands to `run_stt`, it opens a short-lived WS connection, streams the PCM, and
reads the final `{"text": ...}` back.

Pipecat has already done VAD/turn-taking upstream, so `audio` here is one complete
utterance at `sample_rate` (16 kHz) — we just transcribe it and emit one
`TranscriptionFrame`.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

import websockets  # provided by uvicorn[standard] in this venv


class MoonshineSTTService(SegmentedSTTService):
    """One-shot Moonshine transcription over the Vosk WebSocket protocol.

    Segmented, not streaming: Pipecat buffers each VAD-bounded utterance and hands
    the WHOLE thing to run_stt, so Moonshine sees a full phrase ("yes, I'm ready"),
    not 0.1 s frames (which it can't transcribe, giving empty results every time)."""

    def __init__(
        self,
        *,
        url: str = "ws://127.0.0.1:2710",
        language: str = "en",
        sample_rate: int = 16000,
        connect_timeout: float = 4.0,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._url = url
        self._language = language
        self._connect_timeout = connect_timeout

    @property
    def wants_wav_segments(self) -> bool:
        # Our Vosk protocol streams raw 16-bit PCM, so take the segment unwrapped.
        return False

    async def _transcribe_ws(self, audio: bytes) -> str:
        """Send one utterance's PCM to the Moonshine server, return the text.

        Protocol: {"config":{sample_rate}} -> raw PCM16LE -> {"eof":1}; the server
        replies with zero or more {"partial"} then {"text"} finals and a trailing
        {"text":""}. We collect the non-empty finals.
        """
        finals: list[str] = []
        async with websockets.connect(
            self._url, max_size=None, open_timeout=self._connect_timeout, ping_interval=None
        ) as ws:
            await ws.send(json.dumps({"config": {"sample_rate": self.sample_rate}}))
            # Chunk the PCM so a very long utterance streams rather than one huge frame.
            step = self.sample_rate * 2  # ~1 s of 16-bit mono per frame
            for i in range(0, len(audio), step):
                await ws.send(audio[i : i + step])
            await ws.send(json.dumps({"eof": 1}))
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    continue
                try:
                    obj = json.loads(message)
                except Exception:  # noqa: BLE001
                    continue
                if "text" in obj:
                    t = (obj.get("text") or "").strip()
                    if t:
                        finals.append(t)
                    else:
                        break  # trailing empty {"text":""} after eof = done
        return " ".join(finals).strip()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        yield None  # satisfy the async-generator contract before any await
        await self.start_processing_metrics()
        try:
            text = await self._transcribe_ws(audio)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[moonshine-stt] transcription failed ({e}); dropping utterance")
            await self.stop_processing_metrics()
            yield ErrorFrame(f"Moonshine STT error: {e}", fatal=False)
            return
        await self.stop_processing_metrics()
        secs = (len(audio) / 2) / max(1, self.sample_rate)
        if text:
            logger.info(f"[moonshine-stt] heard ({secs:.1f}s): {text!r}")
            yield TranscriptionFrame(text, "", time_now_iso8601(), Language(self._language))
        else:
            # Reached the server but got no words back. If this happens for every
            # utterance while the candidate is clearly speaking, the Moonshine
            # backend (:2710) behind the router at {url} is almost certainly not
            # running — the router accepts the connection but has nothing to proxy to.
            logger.warning(
                f"[moonshine-stt] EMPTY transcription for a {secs:.1f}s utterance via {self._url}. "
                f"If this repeats while you ARE speaking, the Moonshine :2710 backend on the Jitsi box "
                f"is likely down (start it with run-all-asr.sh; check custom-asr/logs/router.log for "
                f"'upstream error')."
            )
