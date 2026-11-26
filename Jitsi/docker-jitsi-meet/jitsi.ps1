<#
.SYNOPSIS
  Convenience wrapper for the Jitsi + per-speaker transcription stack.

.DESCRIPTION
  Combines the base stack (docker-compose.yml), the Jigasi transcriber
  (transcriber.yml), the offline Vosk STT engine (vosk.yml) and the Jibri
  video recorder (jibri.yml).

.EXAMPLE
  .\jitsi.ps1 up           # pull images (first time) and start everything
  .\jitsi.ps1 logs         # follow logs
  .\jitsi.ps1 logs transcriber
  .\jitsi.ps1 ps           # show container status
  .\jitsi.ps1 down         # stop and remove containers
  .\jitsi.ps1 restart
  .\jitsi.ps1 transcripts  # collect saved transcripts into a clean, named folder
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('up', 'down', 'restart', 'logs', 'ps', 'config', 'pull', 'transcripts')]
    [string]$Command = 'up',

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# Transcription is served by the native Moonshine ASR server (../custom-asr),
# reached at ws://host.docker.internal:2700 (set in .env). The bundled kaldi-en
# Vosk container (vosk.yml) is no longer used; start it only to fall back to
# plain Vosk by adding '-f','vosk.yml' here and pointing VOSK_URL at ws://vosk:2700.
# Jibri (jibri.yml) adds headless video recording. Click "Start recording" in a
# call; finished .mp4 files land in .jitsi-meet-cfg/storage/jibri/Video Recordings/.
$files = @('-f', 'docker-compose.yml', '-f', 'transcriber.yml', '-f', 'jibri.yml')

function Invoke-CollectTranscripts {
    # Jigasi saves each meeting's transcript (with per-line timestamps) under
    # .jitsi-meet-cfg/storage/transcripts/<iso>_<uuid>/transcript_*.txt, plus a
    # per-speaker .wav. Those names are UUIDs, so you can't tell which interview
    # is which. This gathers them into a readable, dated folder:
    #   .../transcripts/Interview Transcripts/<room>_<YYYY-MM-DD_HH-MM-SS>.txt
    # Copies only (originals are left in place) and is safe to run repeatedly.
    $root = Join-Path $PSScriptRoot '.jitsi-meet-cfg/storage/transcripts'
    if (-not (Test-Path $root)) { Write-Host "No transcripts yet at $root"; return }
    $dest = Join-Path $root 'Interview Transcripts'
    New-Item -ItemType Directory -Force -Path $dest | Out-Null

    $txts = Get-ChildItem -Path $root -Recurse -Filter 'transcript_*.txt' -File |
        Where-Object { $_.DirectoryName -ne $dest }
    if (-not $txts) { Write-Host "No transcript files found under $root"; return }

    $count = 0
    foreach ($f in $txts) {
        # room name: first line is "... in room <room>@muc.meet.jitsi"
        $firstLine = (Get-Content -Path $f.FullName -TotalCount 1)
        $room = 'meeting'
        if ($firstLine -match 'in room\s+([^@\s]+)') { $room = $Matches[1] }
        # timestamp: the parent folder name begins with an ISO-ish stamp like
        #   2026-11-16T10:56:07.661Z_<uuid>   (newer, with colons) or
        #   2026-10-07T064830.634Z_<uuid>     (older, no colons)
        # Parse by slicing (no regex) so both shapes work: date = first 10 chars,
        # time = the first 6 digits after the 'T'.
        $ts = 'unknown-time'
        try {
            $stamp = ($f.Directory.Name -split '_')[0]   # 2026-...Z
            $datePart = $stamp.Substring(0, 10)          # YYYY-MM-DD
            $tIdx = $stamp.IndexOf('T')
            $timeDigits = ($stamp.Substring($tIdx + 1) -replace '[^0-9]', '')
            if ($datePart -match '^\d{4}-\d{2}-\d{2}$' -and $timeDigits.Length -ge 6) {
                $hh = $timeDigits.Substring(0, 2); $mm = $timeDigits.Substring(2, 2); $ss = $timeDigits.Substring(4, 2)
                $ts = "${datePart}_${hh}-${mm}-${ss}"
            }
        } catch { }
        $safeRoom = ($room -replace '[^A-Za-z0-9._-]', '_')
        $target = Join-Path $dest "${safeRoom}_${ts}.txt"
        # avoid clobbering two meetings that share room+second
        if ((Test-Path $target) -and ((Get-Item $target).Length -ne $f.Length)) {
            $uid = $f.Directory.Name
            if ($uid.Length -gt 8) { $uid = $uid.Substring($uid.Length - 8) }
            $target = Join-Path $dest "${safeRoom}_${ts}_${uid}.txt"
        }
        Copy-Item -Path $f.FullName -Destination $target -Force
        $count++
    }
    Write-Host "Collected $count transcript(s) into: $dest"
    Get-ChildItem -Path $dest -Filter '*.txt' | Sort-Object Name |
        Select-Object -ExpandProperty Name | ForEach-Object { Write-Host "  $_" }
}

switch ($Command) {
    'up'          { docker compose @files up -d @Rest }
    'down'        { docker compose @files down @Rest }
    'restart'     { docker compose @files restart @Rest }
    'logs'        { docker compose @files logs -f --tail=100 @Rest }
    'ps'          { docker compose @files ps @Rest }
    'config'      { docker compose @files config @Rest }
    'pull'        { docker compose @files pull @Rest }
    'transcripts' { Invoke-CollectTranscripts }
}
