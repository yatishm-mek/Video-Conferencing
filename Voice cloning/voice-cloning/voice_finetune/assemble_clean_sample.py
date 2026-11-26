"""Joins the takes from voice_finetune/make_clean_sample.py into one clean recording per model.

For every sentence, Whisper transcribes each take and the one that matches the text
word for word (no cut-off ending, no skipped or repeated words) is kept. Silence at the
edges of each take is trimmed with short fades, so there are no clicks, and sentences are
joined with a natural pause. All models get the same loudness, so they can be compared
fairly.

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_finetune\\assemble_clean_sample.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from voice_finetune.make_clean_sample import PASSAGE, TAKES_DIR  # noqa: E402
from voice_finetune.prepare_dataset import load_whisper  # noqa: E402

OUT_DIR = TAKES_DIR.parent
PAUSE_SECONDS = 0.7         # between paragraphs
EDGE_PAD_SECONDS = 0.06     # silence kept at each edge of a take
FADE_SECONDS = 0.012
SILENCE_DB = -45.0          # below the take's peak: counts as silence when trimming
TARGET_RMS_DB = -20.0
# A take whose last 30 ms are within this many dB of its speech level was cut off mid-sound.
CUT_OFF_WITHIN_DB = 15.0

CONTRACTIONS = {"i'll": "i will", "i'm": "i am", "you're": "you are", "we'll": "we will", "that's": "that is",
                "there's": "there is", "it's": "it is", "what's": "what is", "let's": "let us",
                # Whisper's spelling of a name that is spoken correctly.
                                                "ritika": "rithika", "rithica": "rithika"}


def words(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    out: list[str] = []
    for w in text.split():
        out += CONTRACTIONS.get(w, w).split()
    return out


def word_errors(expected: list[str], heard: list[str]) -> int:
    """Word-level edit distance: skipped, extra or wrong words."""
    row = list(range(len(heard) + 1))
    for i, e in enumerate(expected, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(heard, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (e != h))
    return row[-1]


def db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9)


def frame_db(audio: np.ndarray, sr: int, frame_s: float = 0.02) -> np.ndarray:
    n = max(1, int(frame_s * sr))
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt(np.mean(np.square(frames), axis=1)) + 1e-9)


def trim(audio: np.ndarray, sr: int) -> np.ndarray:
    levels = frame_db(audio, sr)
    loud = np.nonzero(levels > levels.max() + SILENCE_DB)[0]
    if not len(loud):
        return audio
    n = int(0.02 * sr)
    pad = int(EDGE_PAD_SECONDS * sr)
    start, end = max(0, loud[0] * n - pad), min(len(audio), (loud[-1] + 1) * n + pad)
    out = audio[start:end].copy()
    fade = int(FADE_SECONDS * sr)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out


def cut_off(audio: np.ndarray, sr: int) -> bool:
    """True if the very end is still loud speech (the model stopped mid-word)."""
    levels = frame_db(audio, sr)
    speech = levels[levels > levels.max() + SILENCE_DB]
    return db(audio[-int(0.03 * sr):]) > float(np.median(speech)) - CUT_OFF_WITHIN_DB


def to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    n = int(len(audio) * 16000 / sr)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)




def assemble(model_name: str, whisper) -> dict:
    folder = TAKES_DIR / model_name
    pieces, rows, sample_rate = [], [], None
    for i, sentence in enumerate(PASSAGE):
        expected = words(sentence)
        best = None
        for path in sorted(folder.glob(f"s{i:02d}_t*.wav")):
            audio, sr = sf.read(path, dtype="float32")
            audio = audio if audio.ndim == 1 else audio.mean(axis=1)
            segments, _ = whisper.transcribe(to_16k(audio, sr), language="en", beam_size=5,
                                             condition_on_previous_text=False)
            heard = " ".join(s.text.strip() for s in segments)
            errors = word_errors(expected, words(heard))
            clipped = bool(cut_off(audio, sr))
            score = (errors + (2 if clipped else 0), len(audio) / sr)  # then prefer the tighter take
            if best is None or score < best["score"]:
                best = {"score": score, "path": path, "audio": audio, "sr": sr, "heard": heard,
                        "errors": errors, "cut_off": clipped}
        if best is None:
            raise SystemExit(f"no takes for sentence {i + 1} in {folder}")
        sample_rate = sample_rate or best["sr"]
        pieces += [trim(best["audio"], best["sr"]), np.zeros(int(PAUSE_SECONDS * best["sr"]), np.float32)]
        rows.append({"sentence": sentence, "take": best["path"].name, "heard": best["heard"],
                     "word_errors": best["errors"], "cut_off": best["cut_off"]})
        flag = "OK " if best["errors"] == 0 and not best["cut_off"] else "CHECK"
        print(f"  [{flag}] {best['path'].name}: {best['heard']}")

    audio = np.concatenate(pieces[:-1])
    audio *= 10 ** ((TARGET_RMS_DB - db(audio[np.abs(audio) > 1e-4])) / 20)
    audio = np.clip(audio, -0.97, 0.97)
    sf.write(OUT_DIR / f"{model_name}.wav", audio, sample_rate)
    clean = all(r["word_errors"] == 0 and not r["cut_off"] for r in rows)
    report = {"model": model_name, "seconds": round(len(audio) / sample_rate, 1), "sample_rate": sample_rate,
              "all_sentences_clean": clean, "sentences": rows}
    (OUT_DIR / f"{model_name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    whisper = load_whisper()
    for folder in sorted(p for p in TAKES_DIR.iterdir() if p.is_dir()):
        missing = [i + 1 for i in range(len(PASSAGE)) if not list(folder.glob(f"s{i:02d}_t*.wav"))]
        if missing:  # a run that was stopped part-way
            print(f"{folder.name}: skipped, no takes for paragraph(s) {missing}")
            continue
        print(folder.name)
        r = assemble(folder.name, whisper)
        print(f"  -> {OUT_DIR / (folder.name + '.wav')}: {r['seconds']}s, "
              f"{'every sentence clean' if r['all_sentences_clean'] else 'some sentences need another take'}")


if __name__ == "__main__":
    main()
