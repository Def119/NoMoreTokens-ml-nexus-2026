param([string]$RunDirectory = "runs/nested_v7")
$logPath = Join-Path $RunDirectory "training.log"
Write-Host "ML Nexus training progress - Ctrl+C stops this viewer only." -ForegroundColor Cyan
while (-not (Test-Path -LiteralPath $logPath)) { Start-Sleep -Seconds 1 }
Get-Content -LiteralPath $logPath -Tail 25 -Wait
