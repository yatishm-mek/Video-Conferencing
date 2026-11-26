"""
Output guardrails for the ASR transcript.

Removes / masks words or phrases that must never appear in the transcript.
Runs as a post-filter on the decoded text, so it is a hard guarantee regardless
of how the model tokenized the word. Matching is whole-word and case-insensitive,
and multi-word phrases are supported.

Blocklist file format (banned_words.txt): one entry per line, '#' for comments,
blank lines ignored. An entry may be a single word or a multi-word phrase.

Modes:
  mask    -> replace with a mask string (default "***")
  redact  -> replace with "[redacted]"
  remove  -> drop the word entirely
"""

from __future__ import annotations

import re
from pathlib import Path


def load_blocklist(path: Path):
    if not path or not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


class Guardrail:
    def __init__(self, entries, mode: str = "mask", mask: str = "***"):
        self.entries = [e for e in (entries or []) if e.strip()]
        self.mode = mode
        self.mask = mask
        pats = []
        for e in self.entries:
            words = e.strip().split()
            if not words:
                continue
            pats.append(r"(?<![A-Za-z0-9])"
                        + r"\s+".join(re.escape(w) for w in words)
                        + r"(?![A-Za-z0-9])")
        self.re = re.compile("|".join(pats), re.IGNORECASE) if pats else None

    def active(self) -> bool:
        return self.re is not None

    def filter(self, text: str):
        """Return (clean_text, list_of_blocked_matches)."""
        if not self.re or not text:
            return text, []
        found = []

        def repl(m):
            found.append(m.group(0))
            if self.mode == "remove":
                return ""
            if self.mode == "redact":
                return "[redacted]"
            return self.mask

        clean = self.re.sub(repl, text)
        clean = re.sub(r"\s{2,}", " ", clean).strip()
        # tidy space before punctuation left by a removal
        clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
        return clean, found


def from_file(path: Path, mode: str = "mask", mask: str = "***") -> "Guardrail":
    return Guardrail(load_blocklist(path), mode=mode, mask=mask)
