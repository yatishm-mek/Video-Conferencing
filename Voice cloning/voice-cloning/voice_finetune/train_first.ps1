# First fine-tuning run on the 457 clips in data\voice_dataset, logged to output\voice_finetune\first_train.log.
$env:HF_HUB_OFFLINE = "1"
Set-Location (Split-Path -Parent $PSScriptRoot)
& "$env:USERPROFILE\.venvs\qwen3tts\Scripts\python.exe" voice_finetune\train_lora.py --epochs 10 --run_name first 2>&1 |
    Tee-Object -FilePath output\voice_finetune\first_train.log
