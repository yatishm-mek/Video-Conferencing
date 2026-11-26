# Voice Studio

A local web app for the fine-tuned voices (Qwen3-TTS 1.7B, one LoRA per person): Rithika
(`run2` pass 6), Aditi, Sharanya and Suresh (`output\voice_finetune\<name>`, the pass
`voice_finetune\pick_pass.py` suggested unless `voices.json` sets `"epoch"`).

```
& "$env:USERPROFILE\.venvs\qwen3tts\Scripts\python.exe" voice_studio\server.py
```

Then open http://localhost:8097. The base model takes about a minute to load onto the GPU;
switching voice afterwards takes a few seconds (`voice_finetune\voice_bank.py` writes the
chosen person's LoRA into the loaded model instead of loading another 4 GB model).

- **Voices**: pick who speaks; the small play button is that person's real voice. The green
  dot marks the voice currently loaded on the GPU.
- **Hear <name>**: a sample of the fine-tuned voice.
- **Make <name> say anything**: type or pick a prompt; blank lines separate paragraphs.
  Speaking style sets how lively the delivery is (Steady 0.6, Balanced 0.75, Expressive 0.9
  sampling temperature); Natural flow shortens long pauses.
  About 4 s of work per second of speech on the laptop GPU.
- **Compare**: the real voice and the training passes with their similarity scores (and word
  errors for the new voices, pauses for Rithika).
- **Your clips**: everything generated, labelled by voice, saved in `output\voice_studio`
  (WAV + JSON).

Add a voice: train it (see `voice_finetune\README.md`), run `pick_pass.py` for it, and add
an entry to `voices.json` (id, name, run folder, real clip, colour). It appears on the next
page load.

The studio keeps the model on the GPU (about 4.5 GB): stop it (Ctrl+C) before training.
