# Compares Qwen's normal sampling with a calmer one on chosen epochs of a run, to tell whether
# words that come out badly are the model or the randomness of generation.
# Writes output\voice_samples\<run>_epoch<N>.wav (normal) and <run>_epoch<N>_calm.wav, then
# scores every sample's similarity to the real voice.
#
#   .\voice_finetune\test_calm.ps1 -Run run2 -Epochs 5,6,7
param(
    [string]$Run = "run2",
    [int[]]$Epochs = @(5, 6, 7),
    [double]$Temperature = 0.6,          # Qwen's default is 0.9
    [int]$TopK = 30,                     # default 50
    [double]$SubtalkerTemperature = 0.7  # default 0.9
)
$env:HF_HUB_OFFLINE = "1"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = "$env:USERPROFILE\.venvs\qwen3tts\Scripts\python.exe"
$runDir = "output\voice_finetune\$Run"
$takes = "output\voice_samples\takes"

foreach ($e in $Epochs) {
    $model = "$runDir\model-epoch-$e"
    if (-not (Test-Path "$model\model.safetensors")) {
        & $py voice_finetune\export_checkpoint.py $runDir --epoch $e
    }
    if ((Get-ChildItem "$takes\${Run}_epoch$e" -Filter *.wav -ErrorAction SilentlyContinue).Count -lt 15) {
        Write-Host "== $Run epoch $e, normal sampling =="
        & $py voice_finetune\test_voice.py $model --name "${Run}_epoch$e"
    }
    Write-Host "== $Run epoch $e, calm sampling (temperature $Temperature, top_k $TopK, sub-talker $SubtalkerTemperature) =="
    & $py voice_finetune\test_voice.py $model --name "${Run}_epoch${e}_calm" `
        --temperature $Temperature --top_k $TopK --subtalker_temperature $SubtalkerTemperature
}

Write-Host "== checking every take with Whisper and assembling =="
& $py voice_finetune\assemble_clean_sample.py
Write-Host "== similarity to the real voice =="
& $py voice_finetune\similarity.py
Write-Host "done"
