# Fine-tuning the cloned voice (Qwen3-TTS 1.7B, LoRA)

Trains Qwen3-TTS-12Hz-1.7B-Base on recordings of one speaker so it sounds more like
them, on an 8 GB GPU. The base model stays frozen and small LoRA layers are trained
(about 19M parameters, peak about 4.6 GB of GPU memory). The training data, inputs and
loss follow Qwen's official `finetuning/sft_12hz.py`, and the result is a normal Qwen
fine-tuned model with a named speaker (`new_voice`, or set `VOICE_SPEAKER_NAME`), used
with `generate_custom_voice`.

Everything runs in the `qwen3tts` environment:

```
$py = "$env:USERPROFILE\.venvs\qwen3tts\Scripts\python.exe"
```

## 1. Record

Read [reading_script.txt](reading_script.txt) aloud (instructions at the top of the
file) and save the recordings in `data\voice_dataset\recordings` (.wav or .m4a). The
script is general, everyday speech (conversation, stories, explanations, questions,
numbers and names, moods, longer reading-style sentences, food, travel, hobbies, advice
and a few tongue twisters): 470 sentences in 10 sessions, about 4,400 words or 30 minutes
of speech. Reading it twice, on different days, gives about an hour and a more natural
voice.

## 2. Cut the recordings into clips

```
& $py voice_finetune\prepare_dataset.py
& $py voice_finetune\prepare_codes.py
```

`prepare_dataset.py` uses Whisper to split each recording at the pauses, keeps only the
pieces that match a script sentence, and writes 24 kHz clips, `train_raw.jsonl` and
`report.csv` (every piece, kept or dropped, and why). `prepare_codes.py` adds Qwen's audio
codes. The reference clip for every sample is a clean ~10 s clip picked from the
recordings, saved as `ref.wav` (`--ref_audio` to choose another).

## 3. Train

```
& $py voice_finetune\train_lora.py --epochs 10 --run_name first
```

Each epoch saves its LoRA weights to `output\voice_finetune\<run>\epoch-<n>`.

## 4. Export and listen

First, once, the "before" sample: the same voice cloned from its reference clip without
fine-tuning (`ref.wav` is picked from the recordings; the words are its transcript):

```
& $py voice_finetune\make_clean_sample.py qwen3tts_1_7b --ref_audio data\voice_dataset\ref.wav --ref_text "The city was quiet in the early hours, with only the sound of birds and the occasional passing scooter." --name new_voice_zero_shot
```

Then for each epoch worth hearing:

```
& $py voice_finetune\export_checkpoint.py output\voice_finetune\first --epoch 4
& $py voice_finetune\test_voice.py output\voice_finetune\first\model-epoch-4 --name first_epoch4
& $py voice_finetune\assemble_clean_sample.py
```

`test_voice.py` reads the test script (voice_finetune/make_clean_sample.py) three times per
paragraph; `assemble_clean_sample.py` keeps the takes Whisper hears word for word and
writes `output\voice_samples\first_epoch4.wav`, next to `new_voice_zero_shot.wav`
(the same voice before fine-tuning). An exported model is about 4 GB, so export only the
epochs worth hearing.

## More people (one voice each)

Every person reads [reading_script_natural.txt](reading_script_natural.txt) (natural,
conversational speech) and gets their own dataset and their own training run with the
same recipe. Put each person's recordings in `data\voice_datasets\<name>\recordings`, then:

```
.\voice_finetune\train_people.ps1 -People aditi,sharanya,suresh
& $py voice_finetune\pick_pass.py --people aditi sharanya suresh --epochs 5 6 7
```

`train_people.ps1` prepares everyone's clips at once (`data\voice_datasets\<name>`), then
codes and trains one person at a time (`output\voice_finetune\<name>`); logs go to
`output\voice_finetune\logs`. `pick_pass.py` speaks 8 test sentences with each pass, scores
similarity (against that person's real clips, with the other people as the "different
person" reference) and Whisper word errors, and writes
`output\voice_samples\people\<name>\pass<N>.wav` and `scores.json` with a suggested pass.

No export is needed: `voice_bank.py` loads the base model once and switches voices by
writing a person's LoRA and speaker row into it (the same weights `export_checkpoint.py`
would save). The Voice Studio uses it to offer every voice listed in
`voice_studio\voices.json`.

`prepare_dataset.py` treats contractions as the full words when matching ("I am" for
"I'm"), and splits pieces longer than 12 s (lines said without a clear pause) at their
largest gaps. The training transcript stays exactly what was said.
