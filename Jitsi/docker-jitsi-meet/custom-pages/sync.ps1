# Refresh the deployable pages + plans from the single sources, then you redeploy
# custom-pages/ to the Linux Jitsi box (see DEPLOY.md).
#
# Run on Windows after editing launcher/interview.html, or after generating a new
# candidate plan with scripts/make_plan.py.
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$repo = Resolve-Path (Join-Path $here "..\..\..")
$launcher = Join-Path $repo "launcher"

# 1) Publish candidate plans + regenerate launcher/candidates.js (bot-generated questions).
python (Join-Path $repo "integration\build_pages.py")

# 2) Copy the interview page + default plan into custom-pages for deployment.
Copy-Item (Join-Path $launcher "interview.html") (Join-Path $here "interview.html") -Force
Copy-Item (Join-Path $launcher "plan.json")      (Join-Path $here "plan.json")      -Force

Write-Host "Synced interview.html, plan.json and plans/ into custom-pages/. Deploy per DEPLOY.md."
