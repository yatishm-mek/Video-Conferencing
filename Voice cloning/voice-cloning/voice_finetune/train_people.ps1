# Fine-tunes one voice per person, each on their own recordings, with the same recipe as
# Rithika's run2 (LoRA rank 16, 10 passes). Each person's recordings go in
# data\voice_datasets\<name>\recordings. For every person:
#   1. prepare_dataset.py  -> data\voice_datasets\<name>\clips, ref.wav, train_raw.jsonl, report.csv
#   2. prepare_codes.py    -> data\voice_datasets\<name>\train_with_codes.jsonl
#   3. train_lora.py       -> output\voice_finetune\<name>\epoch-0..9
# Step 1 runs for everyone at once (Whisper is small); steps 2-3 run one person at a time,
# because training needs most of the 8 GB GPU. Logs: output\voice_finetune\logs\<name>_<step>.log
#
#   .\voice_finetune\train_people.ps1 -People aditi,sharanya,suresh
#   .\voice_finetune\train_people.ps1 -People suresh -SkipPrepare    # train again on existing clips
param(
    [string[]]$People = @("aditi", "sharanya", "suresh"),
    [string]$Script = "voice_finetune\reading_script_natural.txt",
    [int]$Epochs = 10,
    [switch]$SkipPrepare
)
$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = "$env:USERPROFILE\.venvs\qwen3tts\Scripts\python.exe"
$logs = "output\voice_finetune\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

function Stamp($msg) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) }

if (-not $SkipPrepare) {
    Stamp "preparing clips for: $($People -join ', ')"
    $procs = foreach ($p in $People) {
        $dir = "data\voice_datasets\$p"
        Start-Process -FilePath $py -WorkingDirectory $PWD.Path -NoNewWindow -PassThru `
            -ArgumentList "voice_finetune\prepare_dataset.py", "--recordings", "$dir\recordings", "--script", $Script, "--out_dir", $dir `
            -RedirectStandardOutput "$logs\${p}_prepare.log" -RedirectStandardError "$logs\${p}_prepare.err.log"
    }
    $procs | Wait-Process
    foreach ($i in 0..($People.Count - 1)) {
        $p = $People[$i]
        if (-not (Test-Path "data\voice_datasets\$p\train_raw.jsonl")) { throw "preparing $p failed: see $logs\${p}_prepare.err.log" }
        Stamp "$p clips: $((Get-Content "$logs\${p}_prepare.log" | Select-String '^kept').Line)"
    }
}

function Run($log, [string[]]$arguments) {
    $proc = Start-Process -FilePath $py -WorkingDirectory $PWD.Path -NoNewWindow -PassThru -Wait -ArgumentList $arguments `
        -RedirectStandardOutput "$logs\$log.log" -RedirectStandardError "$logs\$log.err.log"
    if ($proc.ExitCode -ne 0) { throw "$log failed (exit $($proc.ExitCode)): see $logs\$log.err.log" }
}

foreach ($p in $People) {
    $dir = "data\voice_datasets\$p"
    Stamp "$p : converting clips to codes"
    Run "${p}_codes" @("voice_finetune\prepare_codes.py", "--input_jsonl", "$dir\train_raw.jsonl", "--output_jsonl", "$dir\train_with_codes.jsonl")
    Stamp "$p : training ($Epochs passes)"
    Run "${p}_train" @("voice_finetune\train_lora.py", "--train_jsonl", "$dir\train_with_codes.jsonl", "--run_name", $p, "--epochs", $Epochs)
    Stamp "$p : $((Get-Content "$logs\${p}_train.log" | Select-String '^finished').Line)"
}
Stamp "all done"
