# First Summer 1 TQC diagnostic run. Start after the local PolyTrack mod and ghost are ready.
$repoRoot = Split-Path -Parent $PSScriptRoot
$guiPath = Join-Path $repoRoot '.venv\Scripts\polybot-gui.exe'
if (-not (Test-Path -LiteralPath $guiPath)) {
    throw "Install the GUI environment first: $guiPath is missing"
}

$launchArguments = @(
    '--algorithm', 'tqc',
    '--tqc-architecture', 'tiny',
    '--track-name', '"Summer 1"',
    '--device', 'cpu',
    '--seed', '0',
    '--frame-skip', '30',
    '--max-episode-seconds', '60',
    '--timesteps', '100000',
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
    '--tqc-ent-coef', 'auto',
    '--tqc-forward-warmup-fraction', '0.8',
    '--tqc-forward-warmup-steering-std', '0.18',
    '--tqc-initial-throttle-bias', '1.0',
    '--fresh'
)
Start-Process -FilePath $guiPath -ArgumentList $launchArguments `
    -WorkingDirectory $repoRoot -WindowStyle Normal
