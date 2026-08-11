param(
    [int]$TimestepsPerRun = 100000,
    [int]$FrameSkip = 30,
    [int]$MaxSteps = 2000,
    [double]$LearningRate = 0.0009,
    [double]$EntropyCoefficient = 0.005,
    [string]$ModelPath = "models/polybot-real-speed-fs30"
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
        --learning-rate $LearningRate `
        --gamma 0.999 `
        --gae-lambda 0.98 `
        --entropy-coef $EntropyCoefficient `
        --curriculum-last-fraction 0.0 `
        --curriculum-probability 0.0 `
        @resumeArguments `
        --checkpoint-episodes 5 `
        --model-out $ModelPath

    $exitCode = $LASTEXITCODE
    Write-Output "Trainer exited with code $exitCode; restarting in 3 seconds."
    Start-Sleep -Seconds 3
}
