# Install TurboQuantDB git hooks by pointing git at scripts/hooks/.
#
# Usage: pwsh scripts/install_hooks.ps1
# Run from the repo root.

$ErrorActionPreference = "Stop"

$hooksDir = "scripts/hooks"
$hookFile = "$hooksDir/pre-commit"

if (-not (Test-Path $hookFile)) {
    Write-Error "ERROR: $hookFile not found. Run from the repo root."
    exit 1
}

git config core.hooksPath $hooksDir
Write-Host "✓ Git hooks installed (core.hooksPath = $hooksDir)" -ForegroundColor Green
Write-Host "  pre-commit hook will run on every commit."
