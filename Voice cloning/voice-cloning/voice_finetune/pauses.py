"""Measures pauses inside speech (not at the start or end of a clip): how often they happen
and how long they are, for the training clips and for each test sample's chosen takes.
A pause is a stretch within ABOVE_NOISE_DB of the clip's background noise level (its
quietest frames), lasting at least PAUSE_MIN_SECONDS. Natural speech has short breaths between phrases; reading aloud from a
script adds longer, more frequent pauses, and a model trained on it copies them.

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_finetune\\pauses.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from voice_finetune.common import RAW_JSONL  # noqa: E402

FRAME_SECONDS = 0.02
ABOVE_NOISE_DB = 10.0
PAUSE_MIN_SECONDS = 0.25
LONG_PAUSE_SECONDS = 0.5


def inner_pauses(path: Path) -> tuple[list[float], float]:
    """(pause lengths inside the speech, seconds of speech from first to last sound)."""
    audio, sr = sf.read(path, dtype="float32")
    audio = audio if audio.ndim == 1 else audio.mean(axis=1)
    n = int(FRAME_SECONDS * sr)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt(np.mean(np.square(frames), axis=1)) + 1e-9)
    noise = np.percentile(db, 5)
    loud = db > max(noise + ABOVE_NOISE_DB, db.max() - 45)
    idx = np.nonzero(loud)[0]
    if len(idx) < 2:
        return [], 0.0
    inside = loud[idx[0]:idx[-1] + 1]
    pauses, run = [], 0
    for is_loud in inside:
        if is_loud:
            if run * FRAME_SECONDS >= PAUSE_MIN_SECONDS:
                pauses.append(run * FRAME_SECONDS)
            run = 0
        else:
            run += 1
    return pauses, len(inside) * FRAME_SECONDS


def summary(name: str, files: list[Path]) -> dict:
    all_pauses, speech = [], 0.0
    for f in files:
        p, s = inner_pauses(f)
        all_pauses += p
        speech += s
    long = [p for p in all_pauses if p >= LONG_PAUSE_SECONDS]
    r = {"files": len(files), "pauses_per_minute": 60 * len(all_pauses) / speech if speech else 0,
         "long_pauses_per_minute": 60 * len(long) / speech if speech else 0,
         "mean_pause_seconds": float(np.mean(all_pauses)) if all_pauses else 0.0,
         "longest_pause_seconds": max(all_pauses, default=0.0)}
    print(f"{name:34} {r['pauses_per_minute']:7.1f} {r['long_pauses_per_minute']:9.1f} "
          f"{r['mean_pause_seconds']:9.2f} {r['longest_pause_seconds']:9.2f}")
    return r


def main() -> None:
    print(f"{'audio':34} {'pauses':>7} {'long':>9} {'average':>9} {'longest':>9}")
    print(f"{'':34} {'/min':>7} {'(>0.5s)/min':>9} {'(s)':>9} {'(s)':>9}")
    results = {}
    rows = [json.loads(x) for x in RAW_JSONL.read_text(encoding="utf-8").splitlines() if x]
    results["training clips (real voice)"] = summary("training clips (real voice)", [Path(r["audio"]) for r in rows])
    samples = ROOT / "output" / "voice_samples"
    for report in sorted(samples.glob("*.json")):
        try:
            chosen = json.loads(report.read_text(encoding="utf-8"))["sentences"]
        except (KeyError, json.JSONDecodeError):
            continue
        takes = [samples / "takes" / report.stem / s["take"] for s in chosen]
        if all(t.exists() for t in takes):
            results[report.stem] = summary(report.stem, takes)
    (samples / "pauses.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
