$ErrorActionPreference = 'Stop'

# Whisper-backed Vosk-WS ASR server (the Whisper counterpart of run-asr.ps1).
# Speaks the same protocol as the Moonshine server on a different port, so the
# ASR router can forward jigasi to it when Whisper is selected.

$ProjectPath = $PSScriptRoot
Set-Location -Path $ProjectPath

# Find the venv python (Windows or Linux layout).
$VenvPy = Join-Path $ProjectPath ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { $VenvPy = Join-Path $ProjectPath ".venv/bin/python" }
if (-not (Test-Path $VenvPy)) { throw "venv python not found under $ProjectPath\.venv (needs faster-whisper installed)" }

$env:WHISPER_ASR_PORT     = "2701"
$env:WHISPER_MODEL        = "deepdml/faster-whisper-large-v3-turbo-ct2"
$env:WHISPER_DEVICE       = "auto"     # auto | cpu | cuda
$env:WHISPER_COMPUTE_TYPE = "int8"     # int8 on CPU; int8_float16 on CUDA
$env:WHISPER_BEAM_SIZE    = "1"
$env:PYTHONPATH           = Join-Path $ProjectPath "libs"

Write-Host "Starting Whisper ASR server on ws://0.0.0.0:2701 ($env:WHISPER_MODEL)"
& $VenvPy (Join-Path $ProjectPath "whisper_ws_server.py")
if ($LASTEXITCODE -ne 0) { throw "Whisper ASR server exited with code $LASTEXITCODE" }
