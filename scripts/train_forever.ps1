param(
    [int]$TimestepsPerRun = 100000,
    [int]$EpisodesPerRun = 500,
    [int]$FrameSkip = 30,
    [int]$MaxSteps = 2000,
    [double]$LearningRate = 0.0005,
    [double]$EntropyCoefficient = 0.0,
    [double]$GhostPoseReward = 18.0,
    [double]$BarrierPenalty = -50.0,
    [double]$FinishBonus = 1000.0,
    [double]$FinishFastBonus = 2000.0,
    [double]$FinishTargetSeconds = 22.0,
    [double]$FinishPaceDecay = 1.5,
    [double]$GroundSlipPenalty = -1000.0,
    [double]$GroundSlipToleranceDegrees = 5.0,
    [string]$ModelPath = "models/polybot-real-speed-fs30-w128",
    [switch]$QuarterCurriculum,
    [switch]$TimedCurriculum,
    [double]$TimedCurriculumStart = 9.0,
    [double]$TimedCurriculumEnd = 16.0
)

$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $repoRoot ".venv/Scripts/polybot-train.exe"
$modelZip = Join-Path $repoRoot "$ModelPath.zip"
$curriculumSections = @(
    @{ Name = "00-25"; Start = 0.00; End = 0.25 },
    @{ Name = "25-50"; Start = 0.25; End = 0.50 },
    @{ Name = "50-75"; Start = 0.50; End = 0.75 },
    @{ Name = "75-100"; Start = 0.75; End = 1.00 }
)

Set-Location $repoRoot
$rewardArguments = @(
    "--ghost-pose-reward", $GhostPoseReward,
    "--barrier-contact-penalty", $BarrierPenalty,
    "--finish-bonus", $FinishBonus,
    "--finish-fast-bonus", $FinishFastBonus,
    "--finish-target-s", $FinishTargetSeconds,
    "--finish-pace-decay", $FinishPaceDecay,
    "--ground-slip-penalty", $GroundSlipPenalty,
    "--ground-slip-tolerance-deg", $GroundSlipToleranceDegrees
)

if ($TimedCurriculum) {
    $timedSegmentName = "$TimedCurriculumStart-$TimedCurriculumEnd"
    $timedMarker = Join-Path $repoRoot "$ModelPath-curriculum-time-$timedSegmentName.complete"
    while (-not (Test-Path -LiteralPath $timedMarker)) {
        $resumeArguments = @()
        if (Test-Path -LiteralPath $modelZip) {
            $resumeArguments = @("--resume", $ModelPath)
        }
        Write-Output "Training ghost-time segment $TimedCurriculumStart-$TimedCurriculumEnd seconds for $EpisodesPerRun episodes."
        & $trainer `
            --backend websocket `
            --timesteps $TimestepsPerRun `
            --max-episodes $EpisodesPerRun `
            --request-timeout 300 `
            --max-steps $MaxSteps `
            --frame-skip $FrameSkip `
            --learning-rate $LearningRate `
            --gamma 0.999 `
            --gae-lambda 0.98 `
            --entropy-coef $EntropyCoefficient `
            @rewardArguments `
            --curriculum-start-s $TimedCurriculumStart `
            --curriculum-end-s $TimedCurriculumEnd `
            @resumeArguments `
            --checkpoint-episodes 5 `
            --model-out $ModelPath
        if ($LASTEXITCODE -eq 0) {
            New-Item -ItemType File -Path $timedMarker -Force | Out-Null
        } else {
            Write-Output "Timed curriculum exited with code $LASTEXITCODE; restarting in 3 seconds."
            Start-Sleep -Seconds 3
        }
    }
    Write-Output "Completed timed curriculum; reverting to full episodes."
}

if ($QuarterCurriculum) {
    foreach ($section in $curriculumSections) {
        $sectionMarker = Join-Path $repoRoot "$ModelPath-curriculum-$($section.Name).complete"
        while (-not (Test-Path -LiteralPath $sectionMarker)) {
            $resumeArguments = @()
            if (Test-Path -LiteralPath $modelZip) {
                $resumeArguments = @("--resume", $ModelPath)
            }

            Write-Output "Training section $($section.Name)% for $EpisodesPerRun episodes."
            & $trainer `
                --backend websocket `
                --timesteps $TimestepsPerRun `
                --max-episodes $EpisodesPerRun `
                --max-steps $MaxSteps `
                --frame-skip $FrameSkip `
                --learning-rate $LearningRate `
                --gamma 0.999 `
                --gae-lambda 0.98 `
                --entropy-coef $EntropyCoefficient `
                @rewardArguments `
                --curriculum-start-ratio $section.Start `
                --curriculum-end-ratio $section.End `
                @resumeArguments `
                --checkpoint-episodes 5 `
                --model-out $ModelPath

            $phaseExitCode = $LASTEXITCODE
            if ($phaseExitCode -eq 0) {
                New-Item -ItemType File -Path $sectionMarker -Force | Out-Null
                Write-Output "Completed section $($section.Name)%."
            } else {
                Write-Output "Section trainer exited with code $phaseExitCode; restarting in 3 seconds."
                Start-Sleep -Seconds 3
            }
        }
    }

    Write-Output "Completed one-time quarter-track curriculum; reverting to full episodes."
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
        @rewardArguments `
        --curriculum-last-fraction 0.0 `
        --curriculum-probability 0.0 `
        @resumeArguments `
        --checkpoint-episodes 5 `
        --model-out $ModelPath

    $exitCode = $LASTEXITCODE
    Write-Output "Trainer exited with code $exitCode; restarting in 3 seconds."
    Start-Sleep -Seconds 3
}
