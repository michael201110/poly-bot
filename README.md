# PolyBot

PolyBot trains driving policies for [PolyTrack](https://www.kodub.com/apps/polytrack). **GRTQC is the primary learner** for Summer 1. The verified 24.263-second TQC champion is an immutable source policy; legacy TQC and PPO code remain available for reference and controlled experiments. The PolyModLoader bridge targets PolyTrack 0.6.3, with 0.6.2 support. The simulator protocol and model schema remain v2; PolyBot is version 2.3.0.

## Install and run

Python 3.11 or newer and Git LFS are required to use the pinned champion replay.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,train,gui]"
.\.venv\Scripts\polybot-doctor.exe --smoke grtqc
.\.venv\Scripts\polybot-gui.exe
```

Install and load the [PolyModLoader bridge](docs/game-integration.md), select the intended track in the GUI, and leave the game running. The current 0.6.3 bridge includes a configurable [live AI HUD](docs/ai-overlay.md). The GUI does not silently load a track-specific training profile; load the configuration or preset intended for that workspace before starting. Starting from a GRTQC initialization verifies five live deterministic laps against its configured reference before making any RL update. A mismatch stops training and preserves the source checkpoint.

The immutable TQC source is `models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion/`. The historical directory name does not indicate active support for its former algorithm. To recreate the separate GRTQC initialization checkpoint:

```powershell
.\.venv\Scripts\python.exe tools/initialize_grtqc.py --config profiles/training/summer-1-grtqc-causal-30.json --source models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion --destination models/experiments/grtqc-causal-20261002/summer-1/grtqc/initialization
```

The transfer script checks 2,048 saved Summer 1 observations and writes `transfer.json` with the source hash and action errors. The contact-aware reward charges nonterminal wall impacts while filtering touchdown impulses; it starts with fresh replay. Run the shared trainer with the saved configuration:

```powershell
.\.venv\Scripts\polybot-train.exe --config profiles/training/summer-1-grtqc-causal-30.json
```

The first run resumes `models/experiments/grtqc-causal-20261002/summer-1/grtqc/initialization/` automatically. Later runs use `--resume latest` or the GUI's **Continue best model**. New candidate policies are evaluated over five deterministic full laps. Only reliable GRTQC policies with a faster median than the current verified best become champions. This profile continues from the fastest verified policy; a slower lap with fewer contacts does not replace it. Other candidates remain under `checkpoints/step-*-rejected/`. The 22.000-second target is checked from those evaluations.

The current profile trains full laps with a discount horizon that includes the finish, a small entropy coefficient, and a nonterminal contact penalty. Critics first fit discounted driving returns from complete matching-policy episodes, then use ordinary one-step TQC targets. The training distribution is bounded tightly after a live audit showed broader samples breaking the unchanged driver's completion. It preserves the source's 30-tick controls. Earlier curriculum and 64-step profiles remain separate experiments. For unattended continuation with reconnect and a clean stop file:

```powershell
.\.venv\Scripts\python.exe scripts/train_with_stop_file.py --config profiles/training/summer-1-grtqc-causal-30.json --resume models/experiments/grtqc-causal-20261002/summer-1/grtqc/latest --stop-file logs/grtqc-causal-30.stop --retry-transport
```

The current GRTQC profile gives critics 121 real inputs: physical state, PWM state and task clock/reward/failure context. The actor retains its original 105-input matrix plus the four-value PWM contribution, preserving source controls exactly. Raw policy demands are stored before immutable output transforms, so replay also describes the controls resumed on touchdown. Previous 105/109-input replay cannot be padded or reused here. Five paired initialization laps again retained exactly 24.263 seconds with zero action/time drift. The old 24.485-second cleaner candidate remains archived; the source remains unbeaten. See the [causal audit](docs/grtqc-causal-audit.md) for the measured reward conflict, repair evidence and remaining uncertainties.

Creating that stop file requests a saved clean shutdown. Transport retries preserve replay, the remaining step budget and curriculum position; an interrupted initial transfer repeats its fidelity check.

Replay written before reward semantics `nonterminal-contact-v2` must be replaced with `--fresh-replay`. Older contact rewards also charged failed-attempt clawback and early-failure costs even when the car continued; the revised calculation reserves those costs for actual failures. Saved policy weights remain usable.

GRTQC continuation with fresh replay refreezes the saved actor while collecting reliable initial laps and adapting critics to the recollected rewards.

GRTQC warms its newly initialized critics while the transferred actor is frozen. Repeated identical laps during this phase are expected; the readable log labels it **Policy frozen**. This profile requires five complete matching-policy episodes, 4,000 complete-return critic updates, at least 5,000 total critic updates, stable loss/disagreement, and a driving-return calibration error within 20% before unlock. The calibration is an initialization check against recorded returns, not a held-out performance estimate or exact entropy-adjusted TQC value. Subsequent learning uses ordinary TQC updates. See the [training guide](docs/training.md) and [GRTQC experiment record](docs/grtqc-experiment.md) for measured results.

The current profile schedules a learning-actor check every 512 decisions. Once due, policy updates pause while the current attempt ends and critics continue learning, preserving complete-lap rewards in replay. A failed one-lap screen skips the remaining evaluation laps; a finishing policy still needs a full five-lap evaluation for promotion or target confirmation. Five consecutive weaker checks restore the verified actor, keeping roughly the same opportunity for actor updates as the previous three checks spaced 1,000 decisions apart. Frozen checks still use five laps every 5,000 decisions.

Changing the GRTQC return horizon, discount, training variance bound or entropy target on resume retains raw replay and driving weights, then repeats critic warmup for the changed targets. Adding actor controller inputs also repeats warmup without discarding critic weights or replay. Multi-step returns stop at finishes, failures and artificial resets; evaluation or curriculum resets retain bootstrap from the actual final observation. The causal profile retains actual complete evaluation attempts in replay and computes actor gradients on verified-lap states without prescribing actions. Every rejected update returns to the verified pace policy while preserving learned critics and experience. `tools/audit_grtqc_values.py <checkpoint>` compares critic values with recorded completed-lap returns as a diagnostic.

The separate `profiles/training/summer-1-grtqc-contact-20.json` tests more frequent control at 20 ticks per decision. Its transferred source completes 5/5 laps at 24.616 seconds; the 30-tick, 24.263-second reference remains preserved.

To check the local mock environment without the game, run the legacy TQC smoke test or the GRTQC device smoke test:

```powershell
.\.venv\Scripts\polybot-doctor.exe --smoke grtqc
.\.venv\Scripts\polybot-train.exe --algorithm tqc --backend mock --timesteps 2048 --tqc-architecture tiny --tqc-learning-starts 256 --eval-interval 1024 --eval-episodes 2
```

The GUI Status tab and `polybot-live-log logs/<track-slug>/<algorithm>/<run>.jsonl` show readable progress; the JSONL log retains full diagnostics. See [track workspaces](docs/tracks.md) for registry management, migration, and path layout. The [game integration](docs/game-integration.md) and [protocol](docs/protocol.md) explain the simulator connection.

Run `python -m pytest`, `python -m ruff check src tests tools`, `git diff --check`, and `python tools/validate_pml_mod.py` before committing. Community contributions follow the [Code of Conduct](CODE_OF_CONDUCT.md), [Contributing](CONTRIBUTING.md), and [Security](SECURITY.md) policies.

```text
src/polybot/algorithms/    GRTQC, TQC and legacy PPO backends
src/polybot/environment/   simulator environment, observations, rewards, curriculum
src/polybot/training/      typed config, runner, evaluation and metrics
src/polybot/models/        versioned model storage and metadata
src/polybot/gui/           training and evaluation controls
pml-mod/                  PolyModLoader bridge
profiles/                 training and reward recipes
tests/                    protocol, environment, algorithm and GUI tests
```

An independent scratch GRTQC experiment is available through `profiles/training/summer-1-grtqc-scratch-30.json`, with its own output directory, stochastic exploration, adaptive quarter-to-full curriculum and **confirmed sub-23s** milestone. It inherits no TQC weights, replay or output overlays. See [the scratch experiment](docs/grtqc-scratch-experiment.md) for origin checks, reward priorities and parallel simulator setup.
