# Summer 1 TQC GUI run. Use -Resume to continue the latest saved policy and replay.
param([switch]$Resume)
$repoRoot = Split-Path -Parent $PSScriptRoot
$guiPath = Join-Path $repoRoot '.venv\Scripts\polybot-gui.exe'
if (-not (Test-Path -LiteralPath $guiPath)) {
    throw "Install the GUI environment first: $guiPath is missing"
}
$remainingTimesteps = 100000
if ($Resume) {
    $modelPath = Join-Path $repoRoot 'models\summer-1\tqc\latest.zip'
    $metadataPath = Join-Path $repoRoot 'models\summer-1\tqc\latest.metadata.json'
    $replayPath = Join-Path $repoRoot 'models\summer-1\tqc\latest.replay.pkl'
    foreach ($requiredPath in @($modelPath, $metadataPath, $replayPath)) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "TQC resume requires $requiredPath"
        }
    }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    $remainingTimesteps = 100000 - [int]$metadata.training_timesteps
    if ($remainingTimesteps -le 0) {
        throw "The saved TQC model has already reached 100,000 timesteps"
    }
}

$launchArguments = @(
    '--algorithm', 'tqc',
    '--tqc-architecture', 'tiny',
    '--track-name', '"Summer 1"',
    '--device', 'cuda',
    '--seed', '0',
    '--frame-skip', '30',
    '--max-episode-seconds', '60',
    '--timesteps', [string]$remainingTimesteps,
    '--reward-profile', '"Summer 1 - full bootstrap"',
    '--reward-scale', '0.01',
    '--curriculum', 'full',
    '--checkpoint-interval', '25000',
    '--tqc-learning-rate', '0.0003',
    '--tqc-buffer-size', '250000',
    '--tqc-learning-starts', '5000',
    '--tqc-batch-size', '256',
    '--tqc-gamma', '0.999',
    '--tqc-tau', '0.005',
    '--tqc-train-freq', '1',
    '--tqc-gradient-steps', '1',
    '--tqc-ent-coef', 'auto_0.01',
    '--tqc-forward-warmup-fraction', '0.8',
    '--tqc-forward-warmup-steering-std', '0.45',
    '--tqc-initial-throttle-bias', '1.0'
)
if ($Resume) {
    $launchArguments += @('--model', ('"' + $modelPath + '"'), '--resume')
} else {
    $launchArguments += '--fresh'
}
Start-Process -FilePath $guiPath -ArgumentList $launchArguments `
    -WorkingDirectory $repoRoot -WindowStyle Normal
