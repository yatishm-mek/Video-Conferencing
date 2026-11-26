"""Named one-shot timers on asyncio. Starting a timer that is running restarts it."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger


class Timers:
    def __init__(self, on_fire: Callable[[str], Awaitable[None]]):
        self._on_fire = on_fire
        self._tasks: dict[str, asyncio.Task] = {}
        self.durations: dict[str, float] = {}

    def start(self, name: str, secs: float, *, full: float | None = None):
        """Start `name` to fire in `secs`. `full` is its nominal length when
        re-arming for the remainder of a silence window."""
        self.cancel(name)
        self.durations[name] = full if full is not None else secs
        self._tasks[name] = asyncio.get_running_loop().create_task(self._run(name, secs))
        logger.debug(f"[timer] {name} started ({secs:g}s)")

    def cancel(self, name: str):
        task = self._tasks.pop(name, None)
        if task and not task.done():
            task.cancel()

    def cancel_all(self):
        for name in list(self._tasks):
            self.cancel(name)

    def running(self, name: str) -> bool:
        t = self._tasks.get(name)
        return t is not None and not t.done()

    async def _run(self, name: str, secs: float):
        try:
            await asyncio.sleep(secs)
        except asyncio.CancelledError:
            return
        self._tasks.pop(name, None)
        logger.info(f"[timer] {name} fired after {secs:g}s")
        await self._on_fire(name)
