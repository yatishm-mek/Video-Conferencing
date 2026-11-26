"""Adds Qwen's audio codes to the training list, like Qwen's finetuning/prepare_data.py
(Apache-2.0), but with the speech tokenizer from the local base model and UTF-8 files.

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_finetune\\prepare_codes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_finetune.common import BASE_MODEL, CODES_JSONL, RAW_JSONL  # noqa: E402

BATCH = 16


def main() -> None:
    from qwen_tts import Qwen3TTSTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, default=RAW_JSONL)
    parser.add_argument("--output_jsonl", type=Path, default=CODES_JSONL)
    args = parser.parse_args()

    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(BASE_MODEL / "speech_tokenizer"), device_map="cuda:0")
    rows = [json.loads(line) for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line]
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        codes = tokenizer.encode([r["audio"] for r in batch]).audio_codes
        for row, code in zip(batch, codes):
            row["audio_codes"] = code.cpu().tolist()
        print(f"coded {min(i + BATCH, len(rows))}/{len(rows)}", flush=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
