"""Jitsi bridge transport: run the Pipecat bot as a participant in a Jitsi room.

A headless Chromium (Playwright) joins the room with the real lib-jitsi-meet and
exchanges audio with this process over a loopback WebSocket; a custom Pipecat
transport (JitsiBridgeTransport) replaces SmallWebRTCTransport with a one-line
change in controller/pipeline.py.
"""

from interview_bot.transports.jitsi_bridge.transport import (  # noqa: F401
    JitsiBridgeTransport,
    JitsiBridgeInputTransport,
    JitsiBridgeOutputTransport,
)
from interview_bot.transports.jitsi_bridge.client import JitsiBridgeClient  # noqa: F401
