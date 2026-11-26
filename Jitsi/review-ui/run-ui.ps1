$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# Interview Review UI launcher.
#
# Serves a local web page that lists every recorded meeting with its video and a
# click-to-seek transcript. It reads the live Jitsi storage folders, so new
# meetings appear on refresh — no rebuild step.
#
# Defaults: http://127.0.0.1:8080, reading
#   ../docker-jitsi-meet/.jitsi-meet-cfg/storage
# Override with env vars before running, e.g.:
#   $env:REVIEW_UI_PORT = "9000"
#   $env:JITSI_STORAGE  = "D:\path\to\.jitsi-meet-cfg\storage"
# Extra args pass through to server.py (e.g. --port 9000 --host 0.0.0.0).

$py = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python -ErrorAction SilentlyContinue }
if (-not $py) { throw "Python 3 not found on PATH (need python3 or python)." }

$port = if ($env:REVIEW_UI_PORT) { $env:REVIEW_UI_PORT } else { "8080" }
Write-Host "======================================"
Write-Host " Interview Review UI"
Write-Host " Reachable on this machine AND from the network (LAN IP)."
Write-Host " The exact URLs are printed just below. (Ctrl+C to stop)"
Write-Host "======================================"

# Binds to 0.0.0.0 by default so the page opens at this machine's IP, e.g.
# http://192.168.7.174:$port — not just localhost. Set $env:REVIEW_UI_HOST =
# "127.0.0.1" before running to restrict it to this machine only.
& $py.Source "$PSScriptRoot/server.py" @args
