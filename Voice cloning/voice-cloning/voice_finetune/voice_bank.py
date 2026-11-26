"""Several fine-tuned voices from one loaded model, switched in a couple of seconds.

Every voice trained by train_lora.py is the same base model plus small LoRA layers and one
speaker row, so instead of loading a full exported model per voice (about 4 GB, ~35 s),
this loads the base model once and, to switch voice, writes base + (alpha/rank)·B·A into
the changed layers and the voice's speaker embedding into codec row SPEAKER_ROW: exactly
what export_checkpoint.py saves, without the copy on disk. The base model also keeps its
speaker encoder, so voices can be scored (speaker_embedding) in the same process.

    bank = VoiceBank()
    bank.use(RUNS_DIR / "aditi", epoch=6)
    wav, sr = bank.speak("Hello there.")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from voice_finetune.common import BASE_MODEL, SPEAKER_ROW  # noqa: E402

SPEAKER = "voice"  # the one speaker name; whose voice it is depends on the loaded LoRA


class VoiceBank:
    def __init__(self, device: str = "cuda:0") -> None:
        from qwen_tts import Qwen3TTSModel

        self.qwen = Qwen3TTSModel.from_pretrained(str(BASE_MODEL), device_map=device, dtype=torch.bfloat16,
                                                  attn_implementation="sdpa")
        model = self.qwen.model
        # What export_checkpoint.py writes into config.json, set on the loaded model instead.
        model.tts_model_type = "custom_voice"
        talker_config = model.config.talker_config
        talker_config.spk_id = {SPEAKER: SPEAKER_ROW}
        talker_config.spk_is_dialect = {SPEAKER: False}
        model.supported_speakers = talker_config.spk_id.keys()

        self.model = model
        self.device = model.talker.device
        self.params = dict(model.talker.named_parameters())
        self.base: dict[str, torch.Tensor] = {}  # untouched copies (CPU) of every layer a voice changed
        self.current: tuple[str, int] | None = None
        # The base model's own default speaker row, so base() can restore it after a LoRA voice.
        self._base_spk = model.talker.get_input_embeddings().weight[SPEAKER_ROW].detach().to("cpu", copy=True)

    @torch.no_grad()
    def use(self, run_dir: Path, epoch: int) -> None:
        """Switches to <run_dir>/epoch-<epoch> (LoRA) with <run_dir>/speaker_embedding.pt."""
        run_dir = Path(run_dir)
        key = (str(run_dir.resolve()), epoch)
        if key == self.current:
            return
        cfg = json.loads((run_dir / "lora_config.json").read_text(encoding="utf-8"))
        scale = cfg["alpha"] / cfg["rank"]
        lora = load_file(str(run_dir / f"epoch-{epoch}" / "lora.safetensors"))
        changed = set()
        for key_a in [k for k in lora if ".lora_A." in k]:
            name = key_a.split(".lora_A.")[0] + ".weight"
            if name not in self.params:
                raise KeyError(f"the base model has no layer {name}")
            weight = self.params[name]
            if name not in self.base:
                self.base[name] = weight.detach().to("cpu", copy=True)
            a = lora[key_a].to(self.device, torch.float32)
            b = lora[key_a.replace(".lora_A.", ".lora_B.")].to(self.device, torch.float32)
            weight.copy_((self.base[name].to(self.device, torch.float32) + scale * (b @ a)).to(weight.dtype))
            changed.add(name)
        for name in set(self.base) - changed:  # layers the previous voice changed and this one doesn't
            self.params[name].copy_(self.base[name].to(self.device))
        speaker = torch.load(run_dir / "speaker_embedding.pt", map_location="cpu")
        embedding = self.model.talker.get_input_embeddings().weight
        embedding[SPEAKER_ROW] = speaker.to(self.device, embedding.dtype)
        self.current = key

    @torch.no_grad()
    def use_base(self) -> None:
        """Switch to the raw Qwen3-TTS base model: no LoRA, the base default speaker.
        Restores any layers a previous voice changed and the base speaker row.
        (Named use_base, not base, because self.base is the dict of saved layers.)"""
        for name, w in self.base.items():
            self.params[name].copy_(w.to(self.device))
        emb = self.model.talker.get_input_embeddings().weight
        emb[SPEAKER_ROW] = self._base_spk.to(self.device, emb.dtype)
        self.current = None

    def speak(self, text: str, max_new_tokens: int | None = None, **sampling) -> tuple[np.ndarray, int]:
        if max_new_tokens is None:  # a limit from the text, so a bad take can't run on for minutes
            max_new_tokens = int((max(1, len(text.split())) / 2.0 + 3) * 12 * 1.8) + 24
        wavs, sr = self.qwen.generate_custom_voice(text=text, speaker=SPEAKER, language="auto",
                                                   max_new_tokens=max_new_tokens, **sampling)
        return np.asarray(wavs[0], dtype=np.float32).squeeze(), sr

    def speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """The voice fingerprint (unit length) Qwen's speaker encoder hears in audio."""
        import librosa

        target = self.model.speaker_encoder_sample_rate
        if sr != target:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target)
        with torch.inference_mode():
            e = self.model.extract_speaker_embedding(audio=audio.astype(np.float32), sr=target)
        e = e.float().cpu().numpy().reshape(-1)
        return e / np.linalg.norm(e)
