# Voice cloning: Qwen3-TTS 1.7B fine-tuned with LoRA

Four cloned voices (Rithika, Aditi, Sharanya, Suresh), the code that trained them, and a
local web app to use them. Everything runs on one NVIDIA GPU (8 GB is enough).

## Fine-tuned models (included)

Each voice is the base model plus a small LoRA trained on about 25-30 minutes of that
person's recordings (rank 16, alpha 32, 10 passes; pass 6 chosen).

| Voice | Folder | Trained on |
|---|---|---|
| Rithika | `output/voice_finetune/run2/` | 30 min, read aloud from a script |
| Aditi | `output/voice_finetune/aditi/` | 24 min of natural speech |
| Sharanya | `output/voice_finetune/sharanya/` | 28 min of natural speech |
| Suresh | `output/voice_finetune/suresh/` | 26 min of natural speech |

In each folder:

- `epoch-6/lora.safetensors`: the trained LoRA layers (37 MB), the voice itself
- `speaker_embedding.pt`: the voice's speaker embedding (codec row 3000)
- `lora_config.json`, `train_args.json`: the settings used

`data/voice_dataset/ref.wav` and `data/voice_datasets/<name>/ref.wav` are one real ~10 s
clip of each person (the studio plays them for comparison). The full training recordings
are not included.

## Base model (download separately, public)

[Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base),
about 4 GB. Put it in `%USERPROFILE%\tools\voice-tests\qwen3-tts\Qwen3-TTS-12Hz-1.7B-Base`
or set `QWEN3_TTS_BASE` to its folder. For training only, also clone
[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) into
`%USERPROFILE%\tools\voice-tests\Qwen3-TTS-repo` (or set `QWEN3_TTS_REPO`).

## Setup

Python 3.12 with PyTorch for CUDA, then:

```
pip install qwen-tts peft accelerate faster-whisper imageio-ffmpeg librosa soundfile fastapi uvicorn
```

## Use the voices

```
python voice_studio\server.py
```

Then open http://localhost:8097, pick a voice and type. The base model loads once (about
a minute) and voices switch in a few seconds: `voice_finetune/voice_bank.py` writes a
voice's LoRA and speaker embedding into the loaded model, giving exactly the weights a
merged model would have.

In code:

```python
from voice_finetune.voice_bank import VoiceBank
bank = VoiceBank()
bank.use("output/voice_finetune/aditi", epoch=6)
audio, sample_rate = bank.speak("Hello there!", temperature=0.75, top_k=40, subtalker_temperature=0.8)
```

A full standalone model folder for one voice (about 4.3 GB, no separate LoRA needed):

```
python voice_finetune\export_checkpoint.py output\voice_finetune\aditi --epoch 6
```

## Train a new voice

See [voice_finetune/README.md](voice_finetune/README.md): record
`voice_finetune/reading_script_natural.txt` (about 30 minutes), then
`voice_finetune\train_people.ps1` cuts the recordings into clips, trains, and
`voice_finetune\pick_pass.py` compares passes.

| Folder | What it is |
|---|---|
| `voice_finetune/` | data preparation, LoRA training, export, voice bank, tests and scores |
| `voice_studio/` | the web app (FastAPI + a static page); voices are listed in `voices.json` |
| `output/voice_samples/` | Rithika's comparison samples (before fine-tuning, pass 5, pass 6) |

Speed: on an RTX 5050 laptop GPU, 1 second of speech takes about 3-5 seconds to make,
which is fine for clips. Live use needs a faster GPU.
