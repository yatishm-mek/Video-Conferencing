$ErrorActionPreference = 'Stop'

# ASR router: one stable endpoint (ws://0.0.0.0:2700) that jigasi connects to,
# forwarding each connection to the active engine (Moonshine :2710 or Whisper :2701).
# The orchestrator switches the engine per interview via POST http://127.0.0.1:2770/set_engine.
#
# Run Moonshine with ASR_PORT=2710 (so :2700 is free for the router), e.g.:
#   $env:ASR_PORT = "2710"; .\run-asr.ps1

$ProjectPath = $PSScriptRoot
Set-Location -Path $ProjectPath

$VenvPy = Join-Path $ProjectPath ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { $VenvPy = Join-Path $ProjectPath ".venv/bin/python" }
if (-not (Test-Path $VenvPy)) { throw "venv python not found under $ProjectPath\.venv" }

$env:ASR_ROUTER_PORT       = "2700"
$env:ASR_ROUTER_CONTROL_PORT = "2770"
$env:ASR_BACKEND_MOONSHINE = "ws://127.0.0.1:2710"
$env:ASR_BACKEND_WHISPER   = "ws://127.0.0.1:2701"
$env:ASR_DEFAULT_ENGINE    = "moonshine"
$env:PYTHONPATH            = Join-Path $ProjectPath "libs"

Write-Host "Starting ASR router on ws://0.0.0.0:2700 (control http://127.0.0.1:2770)"
& $VenvPy (Join-Path $ProjectPath "asr_router.py")
if ($LASTEXITCODE -ne 0) { throw "ASR router exited with code $LASTEXITCODE" }
