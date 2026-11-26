"""Reads the test script (voice_finetune/make_clean_sample.py's PASSAGE) with a fine-tuned model
from export_checkpoint.py, three takes per paragraph. The takes go to
output/voice_samples/takes/<name>, next to the zero-shot model's, so
assemble_clean_sample.py checks, picks and joins both the same way.

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_finetune\\test_voice.py <model folder> --name finetuned_epoch4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from voice_finetune.full_speed import run_at_full_speed  # noqa: E402
from voice_finetune.make_clean_sample import PASSAGE, TAKES, TAKES_DIR  # noqa: E402
from voice_finetune.common import SPEAKER_NAME  # noqa: E402


def main() -> None:
    from qwen_tts import Qwen3TTSModel

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--name", required=True, help="name of the sample, e.g. finetuned_epoch4")
    parser.add_argument("--takes", type=int, default=TAKES)
    # Sampling: Qwen's defaults (generation_config.json) are temperature 0.9 and top_k 50 for
    # both the talker (sounds, timing, pronunciation) and the sub-talker (fine acoustic
    # detail). Lower is steadier but can sound flatter.
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--subtalker_temperature", type=float)
    args = parser.parse_args()
    sampling = {k: v for k, v in {"temperature": args.temperature, "top_k": args.top_k,
                                  "subtalker_temperature": args.subtalker_temperature}.items() if v is not None}

    run_at_full_speed()
    model = Qwen3TTSModel.from_pretrained(str(args.model_dir), device_map="cuda:0", dtype=torch.bfloat16,
                                          attn_implementation="sdpa")
    out = TAKES_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("s*_t*.wav"):
        old.unlink()
    for i, paragraph in enumerate(PASSAGE):
        for take in range(args.takes):
            t0 = time.perf_counter()
            # "auto": training used no language tag, so this matches it exactly.
            wavs, sample_rate = model.generate_custom_voice(text=paragraph, speaker=SPEAKER_NAME, language="auto",
                                                            **sampling)
            sf.write(out / f"s{i:02d}_t{take}.wav", np.asarray(wavs[0], dtype=np.float32).squeeze(), sample_rate)
            print(f"[{args.name}] paragraph {i + 1}/{len(PASSAGE)} take {take + 1}: {time.perf_counter() - t0:.1f}s",
                  flush=True)


if __name__ == "__main__":
    main()
