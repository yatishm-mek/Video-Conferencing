"""Turns one epoch of train_lora.py into a full model folder in Qwen's fine-tuned format
(the same as sft_12hz.py writes): the LoRA layers merged into the base weights, the voice's
speaker embedding stored as a named speaker, and the config switched to "custom_voice".
Load it with Qwen3TTSModel.from_pretrained(...) and generate_custom_voice(speaker="new_voice")
(the name is common.SPEAKER_NAME, or VOICE_SPEAKER_NAME if set).

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_finetune\\export_checkpoint.py <run folder> --epoch 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from voice_finetune.common import BASE_MODEL, SPEAKER_NAME, SPEAKER_ROW  # noqa: E402


def export(run_dir: Path, epoch: int) -> Path:
    lora_cfg = json.loads((run_dir / "lora_config.json").read_text(encoding="utf-8"))
    scale = lora_cfg["alpha"] / lora_cfg["rank"]
    lora = load_file(str(run_dir / f"epoch-{epoch}" / "lora.safetensors"))
    out = run_dir / f"model-epoch-{epoch}"

    def skip(folder: str, names: list[str]) -> list[str]:
        # The top-level weights are rewritten below; speech_tokenizer/ keeps its own.
        top = Path(folder) == BASE_MODEL
        return [n for n in names if n in (".cache", ".gitattributes") or (top and n == "model.safetensors")]

    shutil.copytree(BASE_MODEL, out, dirs_exist_ok=True, ignore=skip)
    weights = load_file(str(BASE_MODEL / "model.safetensors"))

    # LoRA keys are relative to the talker: "<layer>.lora_A.default.weight" / "<layer>.lora_B.default.weight".
    merged = 0
    for key_a in [k for k in lora if ".lora_A." in k]:
        layer = key_a.split(".lora_A.")[0]
        target = f"talker.{layer}.weight"
        if target not in weights:
            raise SystemExit(f"base model has no weight {target}")
        a = lora[key_a].float()
        b = lora[key_a.replace(".lora_A.", ".lora_B.")].float()
        weights[target] = (weights[target].float() + scale * (b @ a)).to(weights[target].dtype)
        merged += 1

    embedding = weights["talker.model.codec_embedding.weight"]
    speaker = torch.load(run_dir / "speaker_embedding.pt")
    embedding[SPEAKER_ROW] = speaker.to(embedding.dtype)
    for key in [k for k in weights if k.startswith("speaker_encoder")]:  # as sft_12hz.py does
        del weights[key]
    save_file(weights, str(out / "model.safetensors"))

    config = json.loads((BASE_MODEL / "config.json").read_text(encoding="utf-8"))
    config["tts_model_type"] = "custom_voice"
    config.setdefault("talker_config", {})["spk_id"] = {SPEAKER_NAME: SPEAKER_ROW}
    config["talker_config"]["spk_is_dialect"] = {SPEAKER_NAME: False}
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"merged {merged} LoRA layers -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    export(args.run_dir, args.epoch)


if __name__ == "__main__":
    main()
