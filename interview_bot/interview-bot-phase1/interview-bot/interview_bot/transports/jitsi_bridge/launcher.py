"""Launch + supervise the headless Jitsi participant (Playwright Chromium).

Serves the bridge page (bridge/agent.html + worklets) over http://127.0.0.1 so the
browser runs in a secure context (localhost is trusted → getUserMedia + AudioWorklet
work with no flags), then launches headless Chromium pointed at it with the room,
display name and the loopback WS URL. Relaunches the browser if it crashes.

Chromium is audio-only and GPU-disabled so the NVIDIA GPU stays free for STT/TTS/LLM.
Requires Playwright:  pip install playwright  &&  playwright install chromium
"""

from __future__ import annotations

import os
import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from loguru import logger

# EZY-Hire repo root → bridge/ (overridable).
_REPO_ROOT = Path(__file__).resolve().parents[6]
BRIDGE_DIR = Path(os.environ.get("EZY_BRIDGE_DIR", str(_REPO_ROOT / "bridge")))

PUBLIC_URL = os.environ.get("EZY_PUBLIC_URL", "https://192.168.7.174:8443").rstrip("/")

CHROMIUM_ARGS = [
    "--ignore-certificate-errors",              # Jitsi uses a self-signed cert
    "--autoplay-policy=no-user-gesture-required",
    "--use-fake-ui-for-media-stream",           # auto-accept the fake mic
    "--disable-gpu",                            # keep the NVIDIA GPU for STT/TTS/LLM
    "--no-sandbox",
]


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class JitsiBrowserLauncher:
    """Serves the bridge page and supervises the headless Jitsi browser."""

    def __init__(self, *, room: str, display_name: str, ws_port: int, candidate_name: str = ""):
        self._room = room
        self._display_name = display_name
        self._ws_port = ws_port
        self._candidate_name = candidate_name or ""
        self._http_port = _free_port()
        self._httpd: ThreadingHTTPServer | None = None
        self._pw = None
        self._browser = None
        self._page = None
        self._stopped = False

    def _start_http(self):
        handler = partial(SimpleHTTPRequestHandler, directory=str(BRIDGE_DIR))
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self._http_port), handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        logger.info(f"[jitsi-bridge] serving {BRIDGE_DIR} on http://127.0.0.1:{self._http_port}")

    def _page_url(self) -> str:
        from urllib.parse import quote

        return (
            f"http://127.0.0.1:{self._http_port}/agent.html"
            f"?room={quote(self._room)}"
            f"&name={quote(self._display_name)}"
            f"&ws=ws://127.0.0.1:{self._ws_port}"
            f"&domain={quote(PUBLIC_URL)}"
            f"&candidate={quote(self._candidate_name)}"
        )

    async def start(self):
        if not BRIDGE_DIR.is_dir():
            raise RuntimeError(f"bridge dir not found: {BRIDGE_DIR} (set EZY_BRIDGE_DIR)")
        self._start_http()
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=True, args=CHROMIUM_ARGS)
        context = await self._browser.new_context(ignore_https_errors=True)
        self._page = await context.new_page()
        # Surface the bridge page's logs (join progress, errors) so a bot interview
        # can be followed/debugged from the bot's console.
        def _on_console(msg):
            txt = msg.text
            if msg.type in ("error", "warning") or "[bridge]" in txt:
                logger.info(f"[browser] {txt}")
            else:
                logger.debug(f"[browser] {msg.type}: {txt}")
        self._page.on("console", _on_console)
        self._page.on("pageerror", lambda e: logger.warning(f"[browser] pageerror: {e}"))
        url = self._page_url()
        logger.info(f"[jitsi-bridge] opening {url}")
        await self._page.goto(url)

    async def stop(self):
        self._stopped = True
        try:
            if self._browser:
                await self._browser.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._pw:
                await self._pw.stop()
        except Exception:  # noqa: BLE001
            pass
        if self._httpd:
            self._httpd.shutdown()
