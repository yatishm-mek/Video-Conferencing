"""Persistence for controller state: one JSON file per candidate.

Extracted from the live session processor so the storage backend lives in one
place. Today it is an atomic write to data/sessions/<slug>_state.json; swapping
in Redis later means changing only this class.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


class SessionStore:
    """Controller state on disk, saved after every event (Redis later)."""

    def __init__(self, data_dir: Path, candidate: str):
        slug = re.sub(r"[^a-z0-9]+", "_", candidate.lower()).strip("_") or "candidate"
        self.path = Path(data_dir) / "sessions" / f"{slug}_state.json"

    def load(self) -> dict | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save(self, state: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
        tmp.replace(self.path)
