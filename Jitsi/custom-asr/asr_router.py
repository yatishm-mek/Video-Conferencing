"""ASR router — one stable Vosk-WS endpoint that forwards to the active engine.

jigasi is configured once to connect here (ws://<host>:2700). For each new
participant connection the router opens an upstream connection to whichever
engine is currently active (Moonshine or Whisper) and transparently pumps the
Vosk protocol both ways. The active engine is chosen per interview by the
orchestrator via a tiny HTTP control port, so switching Whisper<->Moonshine
needs no jigasi restart.

Topology:
    jigasi ──ws──▶  asr_router  :2700
                        │  (active engine, set via POST :2770/set_engine)
            ┌───────────┴───────────┐
       Moonshine :2710          Whisper :2701
       (run asr_ws_server.py     (run whisper_ws_server.py)
        with ASR_PORT=2710)

Control API (HTTP, :2770):
    POST /set_engine  {"engine": "moonshine" | "whisper"}
    GET  /engine      -> {"engine": "..."}

Run with the same venv as the Moonshine server (needs vendored websockets).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "libs"))
import websockets  # noqa: E402

HOST = os.environ.get("ASR_ROUTER_HOST", "0.0.0.0")
PORT = int(os.environ.get("ASR_ROUTER_PORT", "2700"))
CONTROL_PORT = int(os.environ.get("ASR_ROUTER_CONTROL_PORT", "2770"))

BACKENDS = {
    "moonshine": os.environ.get("ASR_BACKEND_MOONSHINE", "ws://127.0.0.1:2710"),
    "whisper": os.environ.get("ASR_BACKEND_WHISPER", "ws://127.0.0.1:2701"),
}
DEFAULT_ENGINE = os.environ.get("ASR_DEFAULT_ENGINE", "moonshine")

_state_lock = threading.Lock()
_active_engine = DEFAULT_ENGINE if DEFAULT_ENGINE in BACKENDS else "moonshine"


def get_engine() -> str:
    with _state_lock:
        return _active_engine


def set_engine(engine: str) -> bool:
    global _active_engine
    if engine not in BACKENDS:
        return False
    with _state_lock:
        _active_engine = engine
    print(f"[router] active engine -> {engine}")
    return True


# ---------------------------------------------------------------------------
# WebSocket proxy
# ---------------------------------------------------------------------------
async def _pump(src, dst):
    try:
        async for message in src:
            await dst.send(message)
    except websockets.exceptions.ConnectionClosed:
        pass


async def handler(client):
    engine = get_engine()
    upstream_url = BACKENDS[engine]
    peer = f"{getattr(client, 'remote_address', ('?',))[0]}"
    print(f"[router] + {peer} -> {engine} ({upstream_url})")
    try:
        async with websockets.connect(
            upstream_url, max_size=None, ping_interval=20, ping_timeout=None, close_timeout=5,
        ) as upstream:
            await asyncio.gather(
                _pump(client, upstream),
                _pump(upstream, client),
            )
    except Exception as e:  # noqa: BLE001
        print(f"[router] {peer} upstream error ({engine}): {e}")
    finally:
        print(f"[router] - {peer}")


# ---------------------------------------------------------------------------
# HTTP control
# ---------------------------------------------------------------------------
class _Control(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/engine":
            self._send(200, {"engine": get_engine()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/set_engine":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            engine = payload.get("engine", "")
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": f"bad request: {e}"})
            return
        if set_engine(engine):
            self._send(200, {"ok": True, "engine": engine})
        else:
            self._send(400, {"ok": False, "error": f"unknown engine '{engine}'", "choices": list(BACKENDS)})

    def log_message(self, *_args):  # silence default logging
        pass


def _start_control():
    srv = ThreadingHTTPServer((os.environ.get("ASR_ROUTER_CONTROL_HOST", "127.0.0.1"), CONTROL_PORT), _Control)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[router] control on http://127.0.0.1:{CONTROL_PORT} (POST /set_engine)")


async def amain():
    _start_control()
    print(f"[router] engines: {BACKENDS}; default={get_engine()}")
    async with websockets.serve(
        handler, HOST, PORT, max_size=None, ping_interval=20, ping_timeout=None, close_timeout=5,
    ):
        print(f"[router] listening on ws://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[router] stopped.")
