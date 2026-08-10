param(
    [int]$TimestepsPerRun = 100000,
    [int]$FrameSkip = 30,
    [int]$MaxSteps = 3334,
    [string]$ModelPath = "models/polybot-real-fs30"
)

$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $repoRoot ".venv/Scripts/polybot-train.exe"
$modelZip = Join-Path $repoRoot "$ModelPath.zip"

Set-Location $repoRoot

while ($true) {
    $resumeArguments = @()
    if (Test-Path -LiteralPath $modelZip) {
        $resumeArguments = @("--resume", $ModelPath)
    }

    & $trainer `
        --backend websocket `
        --timesteps $TimestepsPerRun `
        --max-steps $MaxSteps `
        --frame-skip $FrameSkip `
        @resumeArguments `
        --checkpoint-episodes 5 `
        --model-out $ModelPath

    $exitCode = $LASTEXITCODE
    Write-Output "Trainer exited with code $exitCode; restarting in 3 seconds."
    Start-Sleep -Seconds 3
}
