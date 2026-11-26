"""Opt-in debugging aids for the live loop.

Everything here is a no-op unless DEBUG_TRACE is on (see config.py), so turning
it off restores the exact previous behaviour. Turning it on gives, for free:

  * a dedicated, rotating ``logs/debug.log`` at DEBUG level (the normal console
    stays at LOG_LEVEL), so a whole run can be inspected after the fact;
  * ``trace(event, **fields)`` - one structured line per interesting event
    (controller actions, turns, timers), easy to grep;
  * ``timed(label)`` - a context manager that logs how long a block took;
  * ``dump(name, obj)`` - a JSON snapshot of a plan / EvalResult / state under
    ``logs/debug/`` to eyeball what the model actually produced.

Usage::

    from interview_bot import debug
    debug.install_debug_logging(settings)      # once, at startup
    debug.trace("user_turn", text=words, silence=0.9)
    with debug.timed("evaluator"):
        ...
    debug.dump("eval_q3", eval_result)
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger

_ENABLED = False
_DUMP_DIR: Path | None = None


def install_debug_logging(settings) -> None:
    """Turn tracing on and add the ``logs/debug.log`` sink when DEBUG_TRACE is set.

    Safe to call when DEBUG_TRACE is off: it returns immediately and nothing
    about the running system changes.
    """
    global _ENABLED, _DUMP_DIR
    _ENABLED = bool(getattr(settings, "DEBUG_TRACE", False))
    if not _ENABLED:
        return
    logs_dir = Path(settings.LOGS_DIR)
    _DUMP_DIR = logs_dir / "debug"
    _DUMP_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        logs_dir / "debug.log",
        level="DEBUG",
        rotation="10 MB",
        retention=3,
        enqueue=True,
    )
    logger.debug(f"[trace] debug logging on -> {logs_dir / 'debug.log'} (snapshots in {_DUMP_DIR})")


def enabled() -> bool:
    return _ENABLED


def trace(event: str, **fields: Any) -> None:
    """Log one structured ``[trace]`` line, only when DEBUG_TRACE is on."""
    if not _ENABLED:
        return
    if fields:
        body = " ".join(f"{k}={_short(v)}" for k, v in fields.items())
        logger.debug(f"[trace] {event} | {body}")
    else:
        logger.debug(f"[trace] {event}")


@contextmanager
def timed(label: str):
    """Log how long the wrapped block took, only when DEBUG_TRACE is on."""
    if not _ENABLED:
        yield
        return
    t0 = time.monotonic()
    try:
        yield
    finally:
        logger.debug(f"[trace] {label} took {(time.monotonic() - t0) * 1000:.0f}ms")


def dump(name: str, obj: Any) -> None:
    """Write a JSON snapshot to ``logs/debug/<name>.json`` for inspection (gated)."""
    if not _ENABLED or _DUMP_DIR is None:
        return
    try:
        payload = obj.model_dump() if hasattr(obj, "model_dump") else obj
        path = _DUMP_DIR / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.debug(f"[trace] dumped {name} -> {path}")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[trace] could not dump {name}: {e}")


def _short(v: Any, limit: int = 100) -> str:
    s = str(v).replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"
