"""Tests a few passes (epochs) of each person's training run and suggests the best one.

For each person the base model is loaded once (voice_bank.py) and every pass speaks the
same test sentences, each with the same random seed, so differences come from the model and
not from luck. Each pass is scored on:
  - similarity: its voice fingerprint against the average of that person's real clips
    (1.0 = identical); "real vs real" and "other people" (the other voices' real clips
    against this one) show what high and low look like;
  - word errors: Whisper's transcript against the sentence, per 100 words.
Suggested pass: fewest word errors, then highest similarity. Trust your ears for the final call.

Writes output\\voice_samples\\people\\<name>\\pass<N>.wav (all sentences joined), real.wav
(a real clip) and scores.json.

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_finetune\\pick_pass.py --people aditi sharanya suresh --epochs 5 6 7
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from voice_finetune.full_speed import run_at_full_speed  # noqa: E402
from voice_finetune.common import RUNS_DIR  # noqa: E402
from voice_finetune.prepare_dataset import load_whisper, words_of  # noqa: E402
from voice_finetune.voice_bank import VoiceBank  # noqa: E402

PEOPLE_DIR = ROOT / "data" / "voice_datasets"
OUT = ROOT / "output" / "voice_samples" / "people"
# Everyday sentences that are not in either reading script.
SENTENCES = [
    "Good morning! I hope your week is off to a great start.",
    "Could you send me the report by Thursday afternoon, please?",
    "The train leaves at half past seven, so we should head out a little early.",
    "Honestly, I didn't expect the movie to be that good, but I really enjoyed it.",
    "First, open the app, then tap settings, and choose the language you prefer.",
    "That's wonderful news, congratulations! You totally deserve it.",
    "It rained all day, so we stayed in, made some tea, and played board games.",
    "Thank you so much for your help today, I really appreciate it.",
]
SAMPLING = {"temperature": 0.75, "top_k": 40, "subtalker_temperature": 0.8}  # the studio's "balanced"
REAL_CLIPS = 40
CHUNK_SECONDS = 6.0
GAP_SECONDS = 0.45


def real_clips(dataset: Path, n: int = REAL_CLIPS) -> list[Path]:
    rows = [json.loads(x) for x in (dataset / "train_raw.jsonl").read_text(encoding="utf-8").splitlines() if x]
    random.seed(0)
    return [Path(r["audio"]) for r in random.sample(rows, min(n, len(rows)))]


def word_errors(expected: list[str], heard: list[str]) -> int:
    ops = difflib.SequenceMatcher(None, expected, heard, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--people", nargs="+", required=True)
    parser.add_argument("--epochs", nargs="+", type=int, default=[5, 6, 7])
    args = parser.parse_args()

    run_at_full_speed()
    bank = VoiceBank()
    whisper = load_whisper()

    def fingerprint(audio: np.ndarray, sr: int) -> np.ndarray:
        """Average of one fingerprint per ~6 s, so long audio is judged like short clips."""
        n = int(CHUNK_SECONDS * sr)
        chunks = [audio[i:i + n] for i in range(0, len(audio), n) if len(audio[i:i + n]) > n // 2] or [audio]
        e = np.mean([bank.speaker_embedding(c, sr) for c in chunks], axis=0)
        return e / np.linalg.norm(e)

    def file_fingerprint(path: Path) -> np.ndarray:
        audio, sr = sf.read(path, dtype="float32")
        return fingerprint(audio, sr)

    # Every voice's real fingerprints, including Rithika's, for the "other people" reference.
    voices = {p.name: p for p in PEOPLE_DIR.iterdir() if (p / "train_raw.jsonl").exists()}
    voices["rithika"] = ROOT / "data" / "voice_dataset"
    real = {name: np.stack([file_fingerprint(c) for c in real_clips(d)]) for name, d in voices.items()}

    for person in args.people:
        run_dir, out = RUNS_DIR / person, OUT / person
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(voices[person] / "ref.wav", out / "real.wav")
        mine = real[person]
        center = mine.mean(axis=0)
        center /= np.linalg.norm(center)
        loo = [float(mine[i] @ (lambda o: o / np.linalg.norm(o))(np.delete(mine, i, 0).mean(0))) for i in range(len(mine))]
        others = {name: float(np.mean(e @ center)) for name, e in real.items() if name != person}
        report = {"person": person, "real_vs_real": float(np.mean(loo)), "other_people": others,
                  "sentences": SENTENCES, "sampling": SAMPLING, "passes": {}}
        print(f"\n== {person}: real vs real {report['real_vs_real']:.3f}; other people "
              + ", ".join(f"{k} {v:.3f}" for k, v in others.items()), flush=True)

        for epoch in args.epochs:
            t0 = time.perf_counter()
            bank.use(run_dir, epoch)
            pieces, errors, words, clean, sr, made, details = [], 0, 0, 0, 24000, 0.0, []
            for i, sentence in enumerate(SENTENCES):
                torch.manual_seed(1000 + i)
                s0 = time.perf_counter()
                audio, sr = bank.speak(sentence, **SAMPLING)
                made += time.perf_counter() - s0
                heard_text = " ".join(s.text.strip() for s in whisper.transcribe(
                    librosa.resample(audio, orig_sr=sr, target_sr=16000), language="en", beam_size=5,
                    condition_on_previous_text=False)[0])
                expected = words_of(sentence)
                e = word_errors(expected, words_of(heard_text))
                errors, words, clean = errors + e, words + len(expected), clean + (e == 0)
                details.append({"sentence": sentence, "heard": heard_text, "word_errors": e,
                                "seconds": round(len(audio) / sr, 2)})
                pieces += [audio, np.zeros(int(GAP_SECONDS * sr), np.float32)]
            joined = np.concatenate(pieces[:-1])
            sf.write(out / f"pass{epoch}.wav", joined, sr)
            speech = sum(d["seconds"] for d in details)
            result = {"similarity": float(fingerprint(joined, sr) @ center),
                      "word_errors_per_100": round(100 * errors / words, 1), "clean_sentences": f"{clean}/{len(SENTENCES)}",
                      "speed_x_realtime": round(speech / made, 2), "details": details}
            report["passes"][str(epoch)] = result
            print(f"{person} pass {epoch}: similarity {result['similarity']:.3f}, word errors "
                  f"{result['word_errors_per_100']}/100, clean {result['clean_sentences']}, "
                  f"{result['speed_x_realtime']}x real time ({time.perf_counter() - t0:.0f}s)", flush=True)

        best = min(report["passes"], key=lambda e: (report["passes"][e]["word_errors_per_100"],
                                                     -report["passes"][e]["similarity"]))
        report["suggested_pass"] = int(best)
        (out / "scores.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{person}: suggested pass {best}", flush=True)


if __name__ == "__main__":
    main()
