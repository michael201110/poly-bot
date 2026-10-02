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

Install and load the [PolyModLoader bridge](docs/game-integration.md), open Summer 1, and leave the game running. The GUI loads the contact-aware GRTQC profile. **Start with these settings** verifies five live deterministic laps against the TQC reference before making any RL update. A mismatch stops training and preserves the source checkpoint.

The immutable TQC source is `models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion/`. The historical directory name does not indicate active support for its former algorithm. To recreate the separate GRTQC initialization checkpoint:

```powershell
.\.venv\Scripts\python.exe tools/initialize_grtqc.py --config profiles/training/summer-1-grtqc-finish-credit-30.json --source models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion --destination models/experiments/grtqc-finish-credit-20261002/summer-1/grtqc/initialization
```

The transfer script checks 2,048 saved Summer 1 observations and writes `transfer.json` with the source hash and action errors. The contact-aware reward charges nonterminal wall impacts while filtering touchdown impulses; it starts with fresh replay. Run the shared trainer with the saved configuration:

```powershell
.\.venv\Scripts\polybot-train.exe --config profiles/training/summer-1-grtqc-finish-credit-30.json
```

The first run resumes `models/experiments/grtqc-finish-credit-20261002/summer-1/grtqc/initialization/` automatically. Later runs use `--resume latest` or the GUI's **Continue best model**. New candidate policies are evaluated over five deterministic full laps. Only reliable GRTQC policies with a faster median than the current verified best become champions. In this profile, reliable cleaner laps within 1.5 seconds of the best are saved separately as `contact-candidate/` and can provide a recovery starting point. Other candidates remain under `checkpoints/step-*-rejected/`. The 22.000-second target is checked from those evaluations.

The current profile trains full laps with a discount horizon that includes the finish, 64-decision reward targets, a small entropy coefficient, and a nonterminal contact penalty. It preserves the source's 30-tick controls. Earlier first-chicane curriculum settings remain as a separate experiment. For unattended training with reconnect and a clean stop file:

```powershell
.\.venv\Scripts\python.exe scripts/train_with_stop_file.py --config profiles/training/summer-1-grtqc-finish-credit-30.json --resume models/experiments/grtqc-finish-credit-20261002/summer-1/grtqc/initialization --stop-file logs/grtqc-finish-credit-30.stop --retry-transport
```

Creating that stop file requests a saved clean shutdown. Transport retries preserve replay, the remaining step budget and curriculum position; an interrupted initial transfer repeats its fidelity check.

Replay written before reward semantics `nonterminal-contact-v2` must be replaced with `--fresh-replay`. Older contact rewards also charged failed-attempt clawback and early-failure costs even when the car continued; the revised calculation reserves those costs for actual failures. Saved policy weights remain usable.

GRTQC continuation with fresh replay refreezes the saved actor while collecting reliable initial laps and adapting critics to the recollected rewards.

GRTQC warms its newly initialized critics while the transferred actor is frozen. Actor updates begin only after the minimum warmup and a stable recent window of quantile loss and critic disagreement. The trainer logs quantile, target, and disagreement statistics in `logs/*.jsonl`. See the [training guide](docs/training.md) and [GRTQC experiment record](docs/grtqc-experiment.md) for the implementation, validation gate, and measured status.

Changing the GRTQC return horizon on resume retains raw replay and driving weights, then repeats critic warmup for the changed targets. Multi-step returns stop at finishes, failures and artificial resets; evaluation or curriculum resets retain bootstrap from the actual final observation. Faster cleaner candidates with the same contact count can replace their previous candidate, while champion promotion still requires beating the verified lap time. `tools/audit_grtqc_values.py <checkpoint>` compares critic values with recorded completed-lap returns as a diagnostic.

The separate `profiles/training/summer-1-grtqc-contact-20.json` tests more frequent control at 20 ticks per decision. Its transferred source completes 5/5 laps at 24.616 seconds; the 30-tick, 24.263-second reference remains preserved.

To check the local mock environment without the game, run the legacy TQC smoke test or the GRTQC device smoke test:

```powershell
.\.venv\Scripts\polybot-doctor.exe --smoke grtqc
.\.venv\Scripts\polybot-train.exe --algorithm tqc --backend mock --timesteps 2048 --tqc-architecture tiny --tqc-learning-starts 256 --eval-interval 1024 --eval-episodes 2
```

The GUI Status tab and `polybot-live-log logs/<run>.jsonl` show readable progress; the JSONL log retains full diagnostics. The [game integration](docs/game-integration.md) and [protocol](docs/protocol.md) explain the simulator connection.

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
