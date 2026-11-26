$ErrorActionPreference = 'Stop'

# Project directory
$ProjectPath = "/home/tekone-ubuntu/Downloads/Jitsi/custom-asr"

Set-Location -Path $ProjectPath

# Python virtual environment
$VenvPy = "$ProjectPath/.venv/bin/python"

if (-not (Test-Path $VenvPy)) {
    throw "Python virtual environment not found at $VenvPy"
}

# Model path
$env:ASR_CHECKPOINT = "$ProjectPath/model"

# ASR configuration
$env:ASR_USE_BIAS = "1"
# End-of-utterance pause before a line is finalized and shown. This is the main
# knob for perceived latency when partials are off: captions appear ~this long
# after you stop talking. 400 ms is responsive without splitting sentences; go
# to 300 for snappier (risks chopping), or 500-600 if lines get cut mid-sentence.
$env:ASR_SILENCE_MS = "400"
$env:ASR_VAD_THRESHOLD = "0.5"
# Flush very long monologues into intermediate lines instead of waiting for a
# full pause. 4000 on CPU: long answers appear in ~4s chunks AND each inference
# runs on <=4s of audio (faster). The original GPU value (15000) makes CPU
# latency much worse, so it stays low here.
$env:ASR_MAX_UTTERANCE_MS = "4000"
# Max tokens per utterance, sized to the 4s window above. 64 is plenty for 4s of
# speech and bounds worst-case latency / repetition loops on CPU.
$env:ASR_MAX_NEW_TOKENS = "64"

# PyTorch CPU threads. This box is a 2-core/4-thread i3 with no AVX2, so
# inference is compute-bound. 0 = auto (cores - 1). Try "2" (physical cores) vs
# "4" to see which is faster on your machine; watch the "[asr] infer … RTF" logs.
$env:ASR_TORCH_THREADS = "0"

# int8 quantization would speed up CPU inference, but the int8 kernels need AVX2,
# which THIS cpu lacks (it would crash with SIGILL). Leave disabled here; set to
# "1" only on a CPU that supports AVX2.
$env:ASR_QUANTIZE = "0"
# Show live interim captions every N ms WHILE a person is still speaking. Each
# partial re-runs the model on the growing utterance, so on a CPU-only machine
# (no GPU here) partials multiply inference load, back up the queue, and cause
# the transcriber to fall behind and DROP audio -> missing/incomplete lines.
# Disabled (0) for completeness. Only raise this above 0 if you move the ASR
# server to a machine with a CUDA GPU.
$env:ASR_PARTIAL_MS = "0"

# Custom libraries
$env:PYTHONPATH = "$ProjectPath/libs"

Write-Host "======================================"
Write-Host "Starting Moonshine ASR Server"
Write-Host "======================================"

Write-Host "Python: $VenvPy"
Write-Host "Model: $env:ASR_CHECKPOINT"
Write-Host "Host: 0.0.0.0"
Write-Host "Port: 2700"

# Start Python server
& $VenvPy "$ProjectPath/asr_ws_server.py"

if ($LASTEXITCODE -ne 0) {
    throw "Moonshine ASR server exited with error code $LASTEXITCODE"
}

