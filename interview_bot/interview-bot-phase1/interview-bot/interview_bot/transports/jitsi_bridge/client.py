"""Loopback WebSocket bridge between the headless Jitsi browser and Pipecat.

The browser (bridge/agent.js) joins the Jitsi room and connects here over a
loopback WebSocket:

    browser -> python : binary  = candidate audio, 48 kHz Int16 mono PCM
                        text    = JSON control {"type": "remote-audio-ready" | "conference-left" | "hello"}
    python  -> browser: binary  = bot TTS audio, 48 kHz Int16 mono PCM
                        text    = JSON control {"type": "flush"}   (barge-in: clear playback)

This client owns the WS server and bridges audio to the Pipecat transport:
incoming candidate PCM is resampled 48 k -> 16 k and yielded as
``InputAudioRawFrame``s; outgoing bot PCM (already 48 k from the output
transport) is queued and shipped without blocking the pipeline.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Awaitable, Callable

from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import InputAudioRawFrame

import websockets  # provided by uvicorn[standard]

BROWSER_SR = 48000  # the browser captures/plays at 48 kHz
PIPELINE_IN_SR = 16000  # Silero VAD + Whisper + the aggregator all run at 16 kHz


class JitsiBridgeClient:
    """Owns the loopback WS server and bridges audio to/from the browser."""

    def __init__(
        self,
        *,
        ws_host: str = "127.0.0.1",
        ws_port: int = 8765,
        on_connected: Callable[[], Awaitable[None]] | None = None,
        on_disconnected: Callable[[], Awaitable[None]] | None = None,
    ):
        self._host = ws_host
        self._port = ws_port
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected

        self._server = None
        self._ws = None  # the single browser connection
        self._started = False
        self._connected = False   # True once the browser reports remote-audio-ready
        self._closing = False

        self._in_queue: asyncio.Queue[InputAudioRawFrame] = asyncio.Queue()
        self._out_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._resampler = create_stream_resampler()
        self._sender_task: asyncio.Task | None = None

    # ---- lifecycle -------------------------------------------------------
    async def connect(self):
        """Start the WS server once (both input and output call this)."""
        if self._started:
            return
        self._started = True
        self._server = await websockets.serve(
            self._handle_browser, self._host, self._port, max_size=None,
            ping_interval=20, ping_timeout=None,
        )
        logger.info(f"[jitsi-bridge] waiting for the browser on ws://{self._host}:{self._port}")

    async def disconnect(self):
        if self._closing:
            return
        self._closing = True
        if self._sender_task:
            self._sender_task.cancel()
            self._sender_task = None
        try:
            if self._ws:
                await self._ws.close()
        except Exception:  # noqa: BLE001
            pass
        if self._server:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # noqa: BLE001
                pass
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and not self._closing

    @property
    def ws_port(self) -> int:
        return self._port

    # ---- browser connection ---------------------------------------------
    async def _handle_browser(self, ws):
        if self._ws is not None:
            logger.warning("[jitsi-bridge] a second browser connected; ignoring it")
            await ws.close()
            return
        self._ws = ws
        logger.info("[jitsi-bridge] browser connected")
        self._sender_task = asyncio.create_task(self._sender_loop())
        try:
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    await self._on_browser_pcm(bytes(message))
                else:
                    await self._on_browser_control(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            logger.info("[jitsi-bridge] browser disconnected")
            self._ws = None
            self._connected = False
            if self._on_disconnected and not self._closing:
                await self._on_disconnected()

    async def _on_browser_pcm(self, pcm48k: bytes):
        # Candidate audio: 48 kHz Int16 -> 16 kHz for the pipeline.
        try:
            pcm16k = await self._resampler.resample(pcm48k, BROWSER_SR, PIPELINE_IN_SR)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[jitsi-bridge] input resample failed (ignored): {e}")
            return
        if pcm16k:
            await self._in_queue.put(
                InputAudioRawFrame(audio=pcm16k, sample_rate=PIPELINE_IN_SR, num_channels=1)
            )

    async def _on_browser_control(self, text: str):
        try:
            obj = json.loads(text)
        except Exception:  # noqa: BLE001
            return
        kind = obj.get("type")
        # The bot starts the interview as soon as a candidate joins the room (even
        # before their mic is flowing). Either signal — a participant joining, or
        # their audio track appearing — begins it, once.
        if kind in ("participant-joined", "remote-audio-ready"):
            who = obj.get("name") or "candidate"
            if not self._connected:
                self._connected = True
                logger.info(f"[jitsi-bridge] {who} joined — bot will begin the interview")
                if self._on_connected:
                    await self._on_connected()
        elif kind == "participant-left":
            # When everyone else has left, end the session.
            if int(obj.get("count", 0)) <= 0:
                logger.info("[jitsi-bridge] candidate left the room")
                self._connected = False
                if self._on_disconnected and not self._closing:
                    await self._on_disconnected()
        elif kind == "conference-left":
            self._connected = False
            if self._on_disconnected and not self._closing:
                await self._on_disconnected()

    async def _sender_loop(self):
        """Ship queued bot audio to the browser without blocking the pipeline."""
        try:
            while True:
                data = await self._out_queue.get()
                if self._ws is None:
                    continue
                try:
                    await self._ws.send(data)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[jitsi-bridge] send failed (ignored): {e}")
        except asyncio.CancelledError:
            pass

    # ---- pipeline-facing API --------------------------------------------
    async def read_audio_frames(self) -> AsyncGenerator[InputAudioRawFrame, None]:
        while True:
            frame = await self._in_queue.get()
            yield frame

    def write_audio_bytes(self, audio: bytes) -> bool:
        """Queue bot PCM (48 kHz Int16) for the browser; returns connection state.

        Non-blocking by design: enqueue and return so the output transport's
        write-timeout watchdog never trips on a slow/absent browser.
        """
        if self.is_connected:
            self._out_queue.put_nowait(audio)
        return self.is_connected

    async def send_control(self, obj: dict[str, Any]):
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps(obj))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[jitsi-bridge] control send failed (ignored): {e}")
