# PolyBot v2

PolyBot trains driving policies for [PolyTrack](https://www.kodub.com/apps/polytrack). PPO, QR-DQN (shown as DQN in the UI), and TQC are equal training modes. All three use the same simulator protocol, observation schema, reward components, curriculum plan, deterministic evaluation, and model registry.

The local mock simulator makes installation and short training checks possible without the game. The real adapter is a PolyModLoader mod targeting PolyTrack 0.6.3, with 0.6.2 support. The wire protocol and model/configuration schemas remain v2; the PolyBot application is version 2.2.0.

## Start

Python 3.11 or later is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,train,gui]"
.\.venv\Scripts\polybot-doctor.exe --smoke tqc
.\.venv\Scripts\polybot-gui.exe
```

In the GUI, choose a track, algorithm, and preset. **Balanced** rewards and algorithm settings are intended as starting points. Hover over any field for a plain-language explanation. **Advanced settings** reveals every algorithm parameter and all reward coefficients. The exact resolved reward values are always visible in the Rewards tab. PPO uses fresh rollouts and PWM steering; DQN uses QR-DQN with nine native digital actions and replay; TQC uses continuous controls and replay. None is universally better.

DQN also has a six-action `no_brake` starting mode. For a staged Summer 1 experiment that learns without brake, transfers the learned Q-network and replay into the nine-action model, then continues training, run `python tools/start_staged_dqn.py`. See the [training guide](docs/training.md) for the transfer threshold and saved model folders.

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

Training saves `models/<track>/<algorithm>/latest/` and a separately proven `champion/`. **Continue best model** prefers the champion when scores tie and restores it after a weaker evaluation. New DQN and TQC champions include replay buffers for exact continuation. Older champions without replay refill a new buffer from the saved policy; TQC attaches that buffer to the champion after a successful pre-training evaluation. A rollback never reuses replay from a failed attempt, and the restored `latest` checkpoint has no evaluation score until tested again. After three consecutive evaluations that lose finishes or substantial track progress, best-model continuation stops; slightly slower complete laps still roll back but do not trigger the stop. **Resume latest (advanced)** keeps the most recent training state even if its evaluation regressed. Changed rewards require fresh replay so stored rewards are never mixed; Continue best handles that automatically for DQN and TQC champions. A lucky finish during stochastic training never replaces the champion. Deterministic full-track evaluation ranks policies by finish rate, progress, then completed lap time. Models and logs are generated files and ignored by Git. There is no v1 model migration.

For the saved Summer 1 TQC champion that predates replay checkpoints, `profiles/training/summer-1-tqc-champion-recovery.json` uses a 20,000-step policy-generated replay refill, learning rate `3e-5`, one gradient update per four environment steps, and a `0.0001` deterministic action drift cap on the champion's evaluated driving path and sampled replay states. The former model lost all five evaluation laps after just 625 reduced-rate updates; the tighter cap preserved all five finishes after 1,250 updates and improved median lap time from 30.29 to 28.98 seconds in the first live check. Further evaluations decide whether an updated policy is promoted.

`profiles/training/summer-1-tqc-20s-pace.json` continues that champion with the `Summer 1 - 20s pace` reward profile. Its valid-finish bonus rises smoothly toward 20 seconds: about 3,833 points at 29 seconds and 6,800 at 20 seconds. Incomplete timeouts now receive the same progress clawback and failure penalty as other failed runs. The changed reward profile triggers a fresh 20,000-step replay refill before learning. In best-model continuation, fully completed laps up to 0.2 seconds slower than the saved champion can keep training; larger regressions restore champion weights and replay. After repeated deterministic first-jump failures at the former `3e-5` learning rate and `0.0001` action drift cap, this profile now uses `1e-5` and `0.00005` for a more gradual continuation.

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
