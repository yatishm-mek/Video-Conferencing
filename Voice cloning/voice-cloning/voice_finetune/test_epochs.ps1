# Makes listening samples for a training run: the voice before fine-tuning (once), then for
# each chosen epoch: export the full model, read the test script, and finally check every take
# with Whisper and join the best ones into output\voice_samples\<run>_epoch<N>.wav.
#
#   .\voice_finetune\test_epochs.ps1 -Run my_run -Epochs 3,6,9
param(
    [string]$Run = "my_run",
    [int[]]$Epochs = @(3, 6, 9)
)
# Each exported model (output\voice_finetune\<run>\model-epoch-<N>) is about 4 GB. Delete the
# ones you don't keep; any epoch can be exported again in a minute from its LoRA weights.
$env:HF_HUB_OFFLINE = "1"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = "$env:USERPROFILE\.venvs\qwen3tts\Scripts\python.exe"
$runDir = "output\voice_finetune\$Run"

if (-not (Test-Path "output\voice_samples\takes\new_voice_zero_shot")) {
    Write-Host "== before fine-tuning: zero-shot clone from the voice's reference clip =="
    $refText = (Get-Content data\voice_dataset\train_raw.jsonl | ConvertFrom-Json |
        Where-Object { $_.audio -eq (Resolve-Path data\voice_dataset\ref.wav).Path } | Select-Object -First 1).text
    if (-not $refText) {
        # ref.wav is a copy of one clip: find the clip with the same audio length
        $refLen = (Get-Item data\voice_dataset\ref.wav).Length
        $refText = (Get-Content data\voice_dataset\train_raw.jsonl | ConvertFrom-Json |
            Where-Object { (Get-Item $_.audio).Length -eq $refLen } | Select-Object -First 1).text
    }
    Write-Host "reference words: $refText"
    & $py voice_finetune\make_clean_sample.py qwen3tts_1_7b --ref_audio data\voice_dataset\ref.wav --ref_text $refText --name new_voice_zero_shot
}

foreach ($e in $Epochs) {
    Write-Host "== $Run epoch $e =="
    & $py voice_finetune\export_checkpoint.py $runDir --epoch $e
    & $py voice_finetune\test_voice.py "$runDir\model-epoch-$e" --name "${Run}_epoch$e"
}

Write-Host "== checking every take with Whisper and assembling =="
& $py voice_finetune\assemble_clean_sample.py
Write-Host "done: listen to output\voice_samples\new_voice_zero_shot.wav and ${Run}_epoch*.wav"
