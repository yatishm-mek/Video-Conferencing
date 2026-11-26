"""How reliably each version pronounces the test script, as numbers.

similarity.py says whose voice it is; this says how clearly it speaks. Whisper transcribes
every take (not only the best one assemble_clean_sample.py keeps) and for each version
reports:
  - word errors: words missing, extra or heard wrong, per 100 words of the script;
  - clean takes: takes Whisper heard word for word;
  - confidence: how sure Whisper was of what it heard (closer to 0 is clearer).
A version that is "hit or miss" shows more word errors and lower confidence.

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_finetune\\clarity.py
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_finetune.make_clean_sample import PASSAGE, TAKES_DIR  # noqa: E402
from voice_finetune.prepare_dataset import load_audio, load_whisper, words_of  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402

SKIP = {"qwen3tts_1_7b"}  # a different person's zero-shot sample, if present


def word_errors(expected: list[str], heard: list[str]) -> int:
    ops = difflib.SequenceMatcher(None, expected, heard, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")


def main() -> None:
    whisper = load_whisper()
    results = {}
    print(f"{'version':22} {'word errors /100':>17} {'clean takes':>12} {'confidence':>11}")
    for folder in sorted(p for p in TAKES_DIR.iterdir() if p.is_dir() and p.name not in SKIP):
        errors = words = clean = takes = 0
        confidence = []
        for i, paragraph in enumerate(PASSAGE):
            expected = words_of(paragraph)
            for take in sorted(folder.glob(f"s{i:02d}_t*.wav")):
                audio = librosa.resample(load_audio(take), orig_sr=24000, target_sr=16000)
                segments = list(whisper.transcribe(audio, language="en", beam_size=5,
                                                   condition_on_previous_text=False)[0])
                heard = words_of(" ".join(s.text for s in segments))
                e = word_errors(expected, heard)
                errors, words, takes = errors + e, words + len(expected), takes + 1
                clean += e == 0
                confidence += [s.avg_logprob for s in segments]
        if not takes or takes < 3 * len(PASSAGE):
            continue  # a run stopped part-way
        results[folder.name] = {"word_errors_per_100": 100 * errors / words, "clean_takes": f"{clean}/{takes}",
                                "confidence": float(np.mean(confidence))}
        r = results[folder.name]
        print(f"{folder.name:22} {r['word_errors_per_100']:17.1f} {r['clean_takes']:>12} {r['confidence']:11.3f}", flush=True)
    (TAKES_DIR.parent / "clarity.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
