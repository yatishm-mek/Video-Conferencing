"""Shortens the reading-style pauses in a test sample, so it flows like natural speech.

The fine-tuned voice copies the pauses of the recordings it learned from (about 20 a
minute, some over a second), which sounds like reading from a script. This is a quick fix
on the finished audio, not on the model: for each paragraph take that
assemble_clean_sample.py chose, Whisper finds the words and every silence between
them is shortened to what natural speech would have there:
  - after a sentence (. ! ?):           at most SENTENCE_PAUSE
  - after a comma, dash or similar:     at most CLAUSE_PAUSE
  - in the middle of a phrase:          at most PHRASE_PAUSE
The start and end of each silence are kept (breaths and word endings stay intact) and the
middle is removed with a short crossfade, so nothing sounds cut. The takes are then joined
like assemble_clean_sample.py does, into output\\voice_samples\\<name>_smooth.wav.

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_finetune\\smooth_pauses.py run2_epoch6 run2_epoch5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from voice_finetune.pauses import inner_pauses  # noqa: E402
from voice_finetune.prepare_dataset import load_whisper  # noqa: E402

SAMPLES = ROOT / "output" / "voice_samples"
SENTENCE_PAUSE = 0.38
CLAUSE_PAUSE = 0.22
PHRASE_PAUSE = 0.10
MIN_GAP = 0.12            # silences shorter than this are left alone
CROSSFADE = 0.015
PARAGRAPH_PAUSE = 0.6
FRAME = 0.01
ABOVE_NOISE_DB = 10.0


def silent_frames(audio: np.ndarray, sr: int) -> np.ndarray:
    n = int(FRAME * sr)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt(np.mean(np.square(frames), axis=1)) + 1e-9)
    return db <= max(np.percentile(db, 5) + ABOVE_NOISE_DB, db.max() - 45)


def silences(audio: np.ndarray, sr: int, start: float, end: float) -> list[tuple[float, float]]:
    """Silent stretches (seconds) between start and end, at least MIN_GAP long."""
    quiet = silent_frames(audio, sr)
    out, run_start = [], None
    for i in range(int(start / FRAME), min(int(end / FRAME), len(quiet))):
        if quiet[i] and run_start is None:
            run_start = i
        elif not quiet[i] and run_start is not None:
            if (i - run_start) * FRAME >= MIN_GAP:
                out.append((run_start * FRAME, i * FRAME))
            run_start = None
    return out


def allowed_pause(words: list, at: float) -> float:
    """How long a pause natural speech would have at time `at`, from the word just before it."""
    before = [w for w in words if w.end <= at + 0.15]
    if not before:
        return PHRASE_PAUSE
    text = before[-1].word.strip()
    if text.endswith((".", "!", "?")):
        return SENTENCE_PAUSE
    if text.endswith((",", ";", ":", "—", "-")):
        return CLAUSE_PAUSE
    return PHRASE_PAUSE


def tighten(audio: np.ndarray, sr: int, words: list) -> tuple[np.ndarray, int, float]:
    """The take with every silence between its first and last word shortened."""
    if not words:
        return audio, 0, 0.0
    cuts = []
    for s, e in silences(audio, sr, words[0].start, words[-1].end):
        keep = allowed_pause(words, s)
        if e - s > keep + 2 * CROSSFADE:
            cuts.append((s + keep / 2, e - keep / 2))  # remove the middle of the silence
    if not cuts:
        return audio, 0, 0.0
    fade = int(CROSSFADE * sr)
    out, pos = [], 0
    for s, e in cuts:
        a, b = int(s * sr), int(e * sr)
        out.append(audio[pos:a])
        pos = b
    out.append(audio[pos:])
    joined = out[0]
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    for piece in out[1:]:  # crossfade each join
        if len(joined) > fade and len(piece) > fade:
            mixed = joined[-fade:] * (1 - ramp) + piece[:fade] * ramp
            joined = np.concatenate([joined[:-fade], mixed, piece[fade:]])
        else:
            joined = np.concatenate([joined, piece])
    return joined, len(cuts), sum(e - s for s, e in cuts)


def edge_trim(audio: np.ndarray, sr: int, words: list) -> np.ndarray:
    lo = max(0, int((words[0].start - 0.08) * sr)) if words else 0
    hi = min(len(audio), int((words[-1].end + 0.15) * sr)) if words else len(audio)
    clip = audio[lo:hi].copy()
    f = int(0.01 * sr)
    clip[:f] *= np.linspace(0, 1, f)
    clip[-f:] *= np.linspace(1, 0, f)
    return clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", help="sample names, e.g. run2_epoch6")
    args = parser.parse_args()

    whisper = load_whisper()
    for name in args.names:
        report = json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8"))
        pieces, sr, removed, n_cuts, before, after = [], None, 0.0, 0, [], []
        for row in report["sentences"]:
            take = SAMPLES / "takes" / name / row["take"]
            audio, sr = sf.read(take, dtype="float32")
            audio = audio if audio.ndim == 1 else audio.mean(axis=1)
            segments, _ = whisper.transcribe(librosa.resample(audio, orig_sr=sr, target_sr=16000), language="en",
                                             word_timestamps=True, condition_on_previous_text=False)
            words = [w for s in segments for w in s.words]
            smooth, cuts, seconds = tighten(audio, sr, words)
            n_cuts, removed = n_cuts + cuts, removed + seconds
            # Kept apart from takes\, which assemble_clean_sample.py reads as takes.
            smooth_take = SAMPLES / "takes_smooth" / name / take.name
            smooth_take.parent.mkdir(parents=True, exist_ok=True)
            sf.write(smooth_take, smooth, sr)
            before += inner_pauses(take)[0]
            after += inner_pauses(smooth_take)[0]
            # Word timings shift after cutting, so trim edges by sound, not by the old timings.
            segments, _ = whisper.transcribe(librosa.resample(smooth, orig_sr=sr, target_sr=16000), language="en",
                                             word_timestamps=True, condition_on_previous_text=False)
            pieces += [edge_trim(smooth, sr, [w for s in segments for w in s.words]),
                       np.zeros(int(PARAGRAPH_PAUSE * sr), np.float32)]
        out = np.concatenate(pieces[:-1])
        out = out / max(1.0, np.abs(out).max() / 0.97)
        sf.write(SAMPLES / f"{name}_smooth.wav", out, sr)
        long_before = sum(p >= 0.5 for p in before)
        long_after = sum(p >= 0.5 for p in after)
        print(f"{name}: shortened {n_cuts} pauses, removed {removed:.1f}s of silence; "
              f"pauses over 0.5s inside paragraphs {long_before} -> {long_after}; "
              f"{report['seconds']}s -> {len(out) / sr:.1f}s; wrote {SAMPLES / (name + '_smooth.wav')}", flush=True)


if __name__ == "__main__":
    main()
