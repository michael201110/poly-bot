param(
    [int]$TimestepsPerRun = 100000,
    [int]$FrameSkip = 30,
    [int]$MaxSteps = 2000,
    [double]$LearningRate = 0.0009,
    [double]$EntropyCoefficient = 0.003,
    [string]$ModelPath = "models/polybot-real-speed-fs30-w128"
)

$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $repoRoot ".venv/Scripts/polybot-train.exe"
$modelZip = Join-Path $repoRoot "$ModelPath.zip"
$lastTenMarker = Join-Path $repoRoot "$ModelPath-last10-50.complete"

Set-Location $repoRoot

while (-not (Test-Path -LiteralPath $lastTenMarker)) {
    $resumeArguments = @()
    if (Test-Path -LiteralPath $modelZip) {
        $resumeArguments = @("--resume", $ModelPath)
    }

    Write-Output "Training the final 10% for 50 episodes."
    & $trainer `
        --backend websocket `
        --timesteps $TimestepsPerRun `
        --max-episodes 50 `
        --max-steps $MaxSteps `
        --frame-skip $FrameSkip `
        --learning-rate $LearningRate `
        --gamma 0.999 `
        --gae-lambda 0.98 `
        --entropy-coef $EntropyCoefficient `
        --curriculum-last-fraction 0.10 `
        --curriculum-probability 1.0 `
        @resumeArguments `
        --checkpoint-episodes 5 `
        --model-out $ModelPath

    $phaseExitCode = $LASTEXITCODE
    if ($phaseExitCode -eq 0) {
        New-Item -ItemType File -Path $lastTenMarker -Force | Out-Null
        Write-Output "Completed 50 final-section episodes; reverting to full episodes."
    } else {
        Write-Output "Final-section trainer exited with code $phaseExitCode; restarting in 3 seconds."
        Start-Sleep -Seconds 3
    }
}

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
