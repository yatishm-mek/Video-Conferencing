"""Fine-tunes Qwen3-TTS-12Hz-1.7B-Base on one speaker with LoRA, so it fits an 8 GB GPU.

Adapted from Qwen's official finetuning/sft_12hz.py (Apache-2.0): same data and the same
loss (first codebook + 0.3 x the other 15), with three fixes so training sees exactly what
generation does (checked on the untrained model: main loss 11.9 -> 1.4, and the end of
speech goes from rank ~450 to the top prediction). sft_12hz.py:
  1. feeds the text embeddings without talker.text_projection, which generation always
     applies (text_projection(get_text_embeddings()(ids)));
  2. shifts the first-codebook labels itself and then passes them to the model, whose
     Hugging Face loss shifts them again (targets two steps ahead), and does the same for
     the 15 sub-codebooks;
  3. gives the sub-codebook predictor the talker state from the frame's own position, but
     generation uses the state from the position before it (the one that predicted the
     frame's first code).
Without these, a fine-tuned model never learns where speech ends and talks on to the
length limit. Other differences:
  - The base model stays frozen; small LoRA layers are trained on every attention and MLP
    projection of the talker and its code predictor (33 blocks).
  - Gradient checkpointing, PyTorch attention (no FlashAttention on Windows), batch 1
    with gradient accumulation.
  - Each epoch saves only the LoRA weights (about 100 MB). export_checkpoint.py turns an
    epoch into a full model in Qwen's format.

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_finetune\\train_lora.py --epochs 10
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, inject_adapter_in_model  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from voice_finetune.full_speed import run_at_full_speed  # noqa: E402
from voice_finetune.common import BASE_MODEL, CODES_JSONL, RUNS_DIR, use_qwen_finetuning_code  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SUB_TALKER_LOSS_WEIGHT = 0.3  # as in sft_12hz.py


def lora_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if "lora_" in k}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", default=str(CODES_JSONL))
    parser.add_argument("--run_name", default=time.strftime("run-%Y%m%d-%H%M"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=8, help="clips per optimizer step")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=0, help="stop early (for a quick test)")
    args = parser.parse_args()

    use_qwen_finetuning_code()
    from dataset import TTSDataset
    from qwen_tts import Qwen3TTSModel

    run_at_full_speed()
    torch.manual_seed(0)
    random.seed(0)
    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    lora_cfg = {"rank": args.rank, "alpha": args.alpha, "targets": LORA_TARGETS}
    (out_dir / "lora_config.json").write_text(json.dumps(lora_cfg, indent=2), encoding="utf-8")

    qwen = Qwen3TTSModel.from_pretrained(str(BASE_MODEL), device_map="cuda:0", dtype=torch.bfloat16,
                                         attn_implementation="sdpa")
    model = qwen.model
    model.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                       target_modules=LORA_TARGETS), model.talker)
    model.talker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    for p in params:
        p.requires_grad_(True)
    print(f"training {sum(p.numel() for p in params) / 1e6:.1f}M LoRA parameters "
          f"of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M", flush=True)

    config = AutoConfig.from_pretrained(str(BASE_MODEL))
    rows = [json.loads(line) for line in Path(args.train_jsonl).read_text(encoding="utf-8").splitlines() if line]
    dataset = TTSDataset(rows, qwen.processor, config)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, total_steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    print(f"{len(rows)} clips, {steps_per_epoch} steps per epoch, {total_steps} steps", flush=True)

    device, speaker_embedding_saved = model.device, False
    step, started = 0, time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                speaker_embedding = model.speaker_encoder(batch["ref_mels"].to(model.dtype)).detach()
                if not speaker_embedding_saved:  # the same reference clip for every sample
                    torch.save(speaker_embedding[0].cpu(), out_dir / "speaker_embedding.pt")
                    speaker_embedding_saved = True

                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                codec_mask = batch["codec_mask"]
                text = model.talker.text_projection(model.talker.get_text_embeddings()(input_ids[:, :, 0]))  # fix 1
                embeddings = (text * batch["text_embedding_mask"]
                              + model.talker.get_input_embeddings()(input_ids[:, :, 1]) * batch["codec_embedding_mask"])
                embeddings[:, 6, :] = speaker_embedding
                for c in range(1, 16):
                    codec_c = model.talker.code_predictor.get_input_embeddings()[c - 1](codec_ids[:, :, c])
                    embeddings = embeddings + codec_c * codec_mask.unsqueeze(-1)

                # Position t has seen inputs 0..t and predicts the first code of the next frame,
                # labels[t + 1]: one shift, done here (fix 2).
                outputs = model.talker(inputs_embeds=embeddings[:, :-1, :], attention_mask=batch["attention_mask"][:, :-1],
                                       output_hidden_states=True)
                logits = outputs.logits.float()
                main_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                            batch["codec_0_labels"][:, 1:].reshape(-1), ignore_index=-100)

                # Sub-codebooks: the state before each frame, then its first code and codes 1..14,
                # predicting codes 1..15 (fix 3; and fix 2, no second shift).
                frames = codec_ids[codec_mask]
                hidden = outputs.hidden_states[0][-1][codec_mask[:, 1:]]
                predictor = model.talker.code_predictor
                sub_inputs = [hidden.unsqueeze(1), model.talker.get_input_embeddings()(frames[:, :1])]
                sub_inputs += [predictor.get_input_embeddings()[c - 1](frames[:, c:c + 1]) for c in range(1, 15)]
                sub_logits = predictor.forward_finetune(inputs_embeds=torch.cat(sub_inputs, dim=1)).logits.float()
                sub_talker_loss = F.cross_entropy(sub_logits.reshape(-1, sub_logits.shape[-1]), frames[:, 1:].reshape(-1))
                loss = main_loss + SUB_TALKER_LOSS_WEIGHT * sub_talker_loss

            (loss / args.grad_accum).backward()
            losses.append(loss.item())
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 or step == 1:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {sum(losses[-args.grad_accum:]) / len(losses[-args.grad_accum:]):.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e} peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                          f"{time.perf_counter() - started:.0f}s", flush=True)
                if args.max_steps and step >= args.max_steps:
                    break

        epoch_dir = out_dir / f"epoch-{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        save_file(lora_state(model.talker), str(epoch_dir / "lora.safetensors"))
        summary = {"epoch": epoch, "mean_loss": sum(losses) / len(losses), "steps": step}
        (epoch_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"epoch {epoch} done: mean loss {summary['mean_loss']:.4f}, saved {epoch_dir}", flush=True)
        if args.max_steps and step >= args.max_steps:
            break

    print(f"finished in {(time.perf_counter() - started) / 60:.1f} min, peak GPU "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
