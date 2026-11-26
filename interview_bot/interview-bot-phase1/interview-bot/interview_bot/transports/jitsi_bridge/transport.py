"""Pipecat transport that bridges the pipeline to a headless Jitsi participant.

Structure mirrors SmallWebRTCTransport (base_input / base_output / base_transport),
but the media moves over the loopback WebSocket in JitsiBridgeClient instead of
aiortc. Swap this in for SmallWebRTCTransport in controller/pipeline.py and the
rest of the pipeline is unchanged.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from interview_bot.transports.jitsi_bridge.client import JitsiBridgeClient


class JitsiBridgeInputTransport(BaseInputTransport):
    """Pushes candidate audio (from the browser, 16 kHz) into the pipeline."""

    def __init__(self, client: JitsiBridgeClient, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._receive_audio_task = None

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        await self._client.connect()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if not self._receive_audio_task and self._params.audio_in_enabled:
            self._receive_audio_task = self.create_task(self._receive_audio())
        await self.set_transport_ready(frame)

    async def _receive_audio(self):
        try:
            async for audio_frame in self._client.read_audio_frames():
                if audio_frame:
                    await self.push_audio_frame(audio_frame)
        except Exception as e:  # noqa: BLE001
            logger.error(f"{self} error receiving audio: {e.__class__.__name__} ({e})")

    async def _teardown(self):
        if self._receive_audio_task:
            await self.cancel_task(self._receive_audio_task)
            self._receive_audio_task = None
        await self._client.disconnect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._teardown()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._teardown()

    async def cleanup(self):
        await super().cleanup()
        await self._teardown()


class JitsiBridgeOutputTransport(BaseOutputTransport):
    """Ships bot audio (48 kHz) to the browser and flushes it on barge-in."""

    def __init__(self, client: JitsiBridgeClient, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        await self._client.connect()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # On an interruption, tell the browser to drop whatever bot audio it has
        # buffered so the candidate stops hearing the bot almost immediately.
        if isinstance(frame, InterruptionFrame):
            await self._client.send_control({"type": "flush"})
        await super().process_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        # Enqueue and return at once (don't await network): the base output
        # transport's write-timeout watchdog must never trip on a slow browser.
        return self._client.write_audio_bytes(frame.audio)

    async def _teardown(self):
        await self._client.disconnect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._teardown()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._teardown()

    async def cleanup(self):
        await super().cleanup()
        await self._teardown()


class JitsiBridgeTransport(BaseTransport):
    """Bridges the pipeline to a headless Jitsi participant via JitsiBridgeClient.

    Event handlers (same names as SmallWebRTCTransport, so pipeline.py is unchanged):
      - on_client_connected(transport, client)
      - on_client_disconnected(transport, client)
    """

    def __init__(
        self,
        client: JitsiBridgeClient,
        params: TransportParams,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params
        self._client = client
        # Route the client's browser events into the transport's event system.
        client._on_connected = self._on_client_connected
        client._on_disconnected = self._on_client_disconnected

        self._input: JitsiBridgeInputTransport | None = None
        self._output: JitsiBridgeOutputTransport | None = None

        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def input(self) -> JitsiBridgeInputTransport:
        if not self._input:
            self._input = JitsiBridgeInputTransport(self._client, self._params, name=self._input_name)
        return self._input

    def output(self) -> JitsiBridgeOutputTransport:
        if not self._output:
            self._output = JitsiBridgeOutputTransport(self._client, self._params, name=self._output_name)
        return self._output

    def get_client_id(self, client: Any) -> str:
        return "jitsi-bridge"

    async def _on_client_connected(self):
        await self._call_event_handler("on_client_connected", self._client)
        if self._input:
            await self._input.push_frame(ClientConnectedFrame())

    async def _on_client_disconnected(self):
        await self._call_event_handler("on_client_disconnected", self._client)
