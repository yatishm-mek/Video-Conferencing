"""Paths and settings shared by the voice fine-tuning scripts (Qwen3-TTS 1.7B, LoRA)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = Path.home() / "tools" / "voice-tests"
BASE_MODEL = Path(os.getenv("QWEN3_TTS_BASE", TOOLS / "qwen3-tts" / "Qwen3-TTS-12Hz-1.7B-Base"))
# Qwen's official fine-tuning code (dataset.py), from github.com/QwenLM/Qwen3-TTS.
QWEN_REPO = Path(os.getenv("QWEN3_TTS_REPO", TOOLS / "Qwen3-TTS-repo"))

DATA_DIR = ROOT / "data" / "voice_dataset"
CLIPS_DIR = DATA_DIR / "clips"           # 24 kHz mono training clips
REF_AUDIO = DATA_DIR / "ref.wav"         # one reference clip for every sample, as Qwen recommends
RAW_JSONL = DATA_DIR / "train_raw.jsonl"
CODES_JSONL = DATA_DIR / "train_with_codes.jsonl"
RUNS_DIR = ROOT / "output" / "voice_finetune"

# The fine-tuned voice's name in the exported model (generate_custom_voice(speaker=...)).
SPEAKER_NAME = os.getenv("VOICE_SPEAKER_NAME", "new_voice")
SAMPLE_RATE = 24000                      # Qwen3-TTS works at 24 kHz
# The fine-tuned voice is stored in this row of the codec embedding, as in Qwen's sft_12hz.py.
SPEAKER_ROW = 3000


def use_qwen_finetuning_code() -> None:
    sys.path.insert(0, str(QWEN_REPO / "finetuning"))
