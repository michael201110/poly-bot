# Summer 1 TQC GUI run. Use -Resume or -ModelPath for a saved policy and replay.
param([switch]$Resume, [string]$ModelPath)
$repoRoot = Split-Path -Parent $PSScriptRoot
$guiPath = Join-Path $repoRoot '.venv\Scripts\polybot-gui.exe'
if (-not (Test-Path -LiteralPath $guiPath)) {
    throw "Install the GUI environment first: $guiPath is missing"
}
$targetTimesteps = 4000000 # About 24 hours at the observed 45-50 steps/second.
$remainingTimesteps = $targetTimesteps
$settings = @{
    architecture = 'tiny'; learning_rate = 0.00015; buffer_size = 500000
    learning_starts = 5000; batch_size = 256; gamma = 0.999; tau = 0.005
    train_freq = 2; gradient_steps = 1; ent_coef = 'auto_0.005'
    forward_warmup_fraction = 0.8; forward_warmup_steering_std = 0.45
    initial_throttle_bias = 1.0; forward_prior_initial = 0.2
    forward_prior_steps = 4000000
}
if ($Resume -or $ModelPath) {
    if (-not $ModelPath) {
        $ModelPath = Join-Path $repoRoot 'models\summer-1\tqc\latest.zip'
    }
    $metadataPath = [System.IO.Path]::ChangeExtension($ModelPath, '.metadata.json')
    $replayPath = [System.IO.Path]::ChangeExtension($ModelPath, '.replay.pkl')
    foreach ($requiredPath in @($ModelPath, $metadataPath, $replayPath)) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "TQC resume requires $requiredPath"
        }
    }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    foreach ($key in @($settings.Keys)) {
        $saved = $metadata.tqc_hyperparameters.PSObject.Properties[$key]
        if ($saved) { $settings[$key] = $saved.Value }
    }
    $remainingTimesteps = $targetTimesteps - [int]$metadata.training_timesteps
    if ($remainingTimesteps -le 0) {
        throw "The saved TQC model has already reached $targetTimesteps timesteps"
    }
}

$launchArguments = @(
    '--algorithm', 'tqc',
    '--tqc-architecture', [string]$settings.architecture,
    '--track-name', '"Summer 1"',
    '--device', 'cuda',
    '--seed', '0',
    '--frame-skip', '30',
    '--max-episode-seconds', '60',
    '--timesteps', [string]$remainingTimesteps,
    '--reward-profile', '"Summer 1 - TQC stable"',
    '--reward-scale', '0.01',
    '--curriculum', 'full',
    '--checkpoint-interval', '250000',
    '--tqc-learning-rate', ([string]$settings.learning_rate),
    '--tqc-buffer-size', ([string]$settings.buffer_size),
    '--tqc-learning-starts', ([string]$settings.learning_starts),
    '--tqc-batch-size', ([string]$settings.batch_size),
    '--tqc-gamma', ([string]$settings.gamma),
    '--tqc-tau', ([string]$settings.tau),
    '--tqc-train-freq', ([string]$settings.train_freq),
    '--tqc-gradient-steps', ([string]$settings.gradient_steps),
    '--tqc-ent-coef', ([string]$settings.ent_coef),
    '--tqc-forward-warmup-fraction', ([string]$settings.forward_warmup_fraction),
    '--tqc-forward-warmup-steering-std', ([string]$settings.forward_warmup_steering_std),
    '--tqc-initial-throttle-bias', ([string]$settings.initial_throttle_bias),
    '--tqc-forward-prior-initial', ([string]$settings.forward_prior_initial),
    '--tqc-forward-prior-steps', ([string]$settings.forward_prior_steps)
)
if ($Resume -or $ModelPath) {
    $launchArguments += @('--model', ('"' + $ModelPath + '"'), '--resume')
} else {
    $launchArguments += '--fresh'
}
Start-Process -FilePath $guiPath -ArgumentList $launchArguments `
    -WorkingDirectory $repoRoot -WindowStyle Normal
