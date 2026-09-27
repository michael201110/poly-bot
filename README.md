# PolyBot v2

PolyBot trains driving policies for [PolyTrack](https://www.kodub.com/apps/polytrack). PPO, DQN, and TQC are equal training modes. All three use the same simulator protocol, observation schema, reward components, curriculum plan, deterministic evaluation, and model registry.

The local mock simulator makes installation and short training checks possible without the game. The real adapter is a PolyModLoader mod targeting PolyTrack 0.6.3, with 0.6.2 support. The wire protocol and model/configuration schemas remain v2; the PolyBot application is version 2.1.0.

## Start

Python 3.11 or later is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,train,gui]"
.\.venv\Scripts\polybot-doctor.exe --smoke tqc
.\.venv\Scripts\polybot-gui.exe
```

In the GUI, choose a track, algorithm, and preset. **Balanced** rewards and algorithm settings are intended as starting points. Hover over any field for a plain-language explanation. **Advanced settings** reveals every algorithm parameter and all reward coefficients. The exact resolved reward values are always visible in the Rewards tab. PPO uses fresh rollouts and PWM steering; DQN uses nine native digital actions and replay; TQC uses continuous controls and replay. None is universally better.

To try a short run on the local mock:

```powershell
.\.venv\Scripts\polybot-train.exe --algorithm ppo --backend mock --timesteps 2048 --ppo-architecture tiny --ppo-rollout 256 --ppo-batch 64 --eval-interval 1024 --eval-episodes 2
.\.venv\Scripts\polybot-train.exe --algorithm dqn --backend mock --timesteps 2048 --dqn-architecture tiny --dqn-learning-starts 256 --eval-interval 1024 --eval-episodes 2
.\.venv\Scripts\polybot-train.exe --algorithm tqc --backend mock --timesteps 2048 --tqc-architecture tiny --tqc-learning-starts 256 --eval-interval 1024 --eval-episodes 2
```

To train against the game, install the [PolyModLoader bridge](docs/game-integration.md), load a track and ghost, then choose **WebSocket** in the GUI or run:

```powershell
.\.venv\Scripts\polybot-train.exe --algorithm tqc --backend websocket --track-name "Summer 1" --track-id current --frame-skip 30 --timesteps 100000 --reward-profile Balanced
```

Training saves `models/<track>/<algorithm>/latest/` and a separately proven `champion/`. DQN and TQC latest include replay buffers for resume; champion playback only needs the policy. Changing reward definitions during a replay-based run would leave old reward values in replay, so resume rejects it. A lucky finish during stochastic training never replaces the champion. Deterministic full-track evaluation ranks policies by finish rate, progress, then completed lap time. Models and logs are generated files and ignored by Git. There is no v1 model migration.

The GUI Status tab shows short episode and evaluation summaries. Full reward diagnostics stay in the run's JSONL file. To follow a running log in a separate readable window, run `polybot-live-log logs/<run>.jsonl`. Use `polybot-live-log "logs/summer-1-tqc-*.jsonl" --follow-newest` to switch automatically when a new run starts. Closing this window does not stop training.

```powershell
.\.venv\Scripts\polybot-eval.exe --algorithm tqc --track-name "Summer 1" --slot champion --backend websocket --episodes 5
.\.venv\Scripts\polybot-drive.exe --algorithm tqc --track-name "Summer 1" --slot champion --backend websocket --realtime
```

See [training](docs/training.md), [game integration](docs/game-integration.md), and the [digital simulator protocol](docs/protocol.md). Validate changes with `python -m pytest`, `python -m ruff check .`, and `python tools/validate_pml_mod.py`.

## Repository

```text
src/polybot/control/       Digital and PWM action adapters
src/polybot/environment/   Gym environment, observations, rewards, curriculum
src/polybot/algorithms/    PPO, DQN and TQC backends and registry
src/polybot/training/      Typed config, runner, evaluation, metrics, devices
src/polybot/models/        v2 storage and metadata
src/polybot/gui/           Guided basic settings and complete advanced settings
pml-mod/                  PolyModLoader adapter for PolyTrack 0.6.2/0.6.3
profiles/rewards/          Named reward recipes
tests/                     Protocol, environment, backend, GUI and device tests
```
