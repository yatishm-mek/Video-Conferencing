"""Makes takes for a clean, uncut sample of the cloned voice, for judging quality only
(speed doesn't matter here). Each paragraph of PASSAGE is generated whole, without
streaming, several times; voice_finetune/assemble_clean_sample.py then keeps the take of each
paragraph that Whisper hears word for word and joins them into one recording.

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_finetune\\make_clean_sample.py qwen3tts_1_7b --ref_audio <clip> --ref_text "<its words>"

Pass a clip of the voice, the exact words in it, and a sample name, e.g. the fine-tuning
voice before training:

    ... voice_finetune\\make_clean_sample.py qwen3tts_1_7b --ref_audio data\\voice_dataset\\ref.wav
        --ref_text "The city was quiet in the early hours, ..." --name new_voice_zero_shot
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

from voice_finetune.common import BASE_MODEL  # noqa: E402
from voice_finetune.full_speed import run_at_full_speed  # noqa: E402

TAKES_DIR = ROOT / "output" / "voice_samples" / "takes"
TAKES = 3  # cloning is random: a few takes per paragraph, the best one is kept

# The test script, one paragraph per entry: a paragraph is spoken in one go, so it keeps
# its natural flow (pace, pitch and pauses between its sentences).
PASSAGE = [
    "Hi, I'm Rithika!",
    "Today, I want to talk about something that is transforming the world around us — Artificial Intelligence.",
    "AI is no longer just a futuristic concept. It's already part of our everyday lives. From voice assistants "
    "and chatbots to self-driving cars and smart healthcare systems, AI is making things faster, smarter, and "
    "more efficient.",
    "One of the most exciting things about AI is its ability to learn from data, understand patterns, and help "
    "people solve complex problems.",
    "But I believe AI should not replace human creativity. Instead, it should work alongside us, helping us "
    "achieve things that were once difficult or impossible.",
    "The future of AI is full of possibilities, and I'm excited to see where this technology takes us.",
    "Thank you for listening!",
]


def qwen3tts_1_7b(ref_audio: Path, ref_text: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(str(BASE_MODEL), device_map="cuda:0",
                                          dtype=torch.bfloat16, attn_implementation="sdpa")
    voice = model.create_voice_clone_prompt(ref_audio=str(ref_audio), ref_text=ref_text)

    def speak(text: str):
        wavs, sample_rate = model.generate_voice_clone(text=text, language="English", voice_clone_prompt=voice)
        return wavs[0], sample_rate

    return speak


MODELS = {"qwen3tts_1_7b": qwen3tts_1_7b}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--ref_audio", type=Path, required=True, help="clip of the voice to clone")
    parser.add_argument("--ref_text", required=True, help="exact words spoken in --ref_audio")
    parser.add_argument("--name", help="name of the sample (default: the model's name)")
    args = parser.parse_args()

    run_at_full_speed()
    out = TAKES_DIR / (args.name or args.model)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("s*_t*.wav"):  # takes of an earlier script
        old.unlink()
    speak = MODELS[args.model](args.ref_audio, args.ref_text)
    speak("Warming up the voice.")
    for i, sentence in enumerate(PASSAGE):
        for take in range(TAKES):
            t0 = time.perf_counter()
            audio, sample_rate = speak(sentence)
            path = out / f"s{i:02d}_t{take}.wav"
            sf.write(path, np.asarray(audio, dtype=np.float32).squeeze(), sample_rate)
            print(f"[{out.name}] paragraph {i + 1}/{len(PASSAGE)} take {take + 1}: "
                  f"{time.perf_counter() - t0:.1f}s -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
