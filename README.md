# poly-bot

A reinforcement-learning environment and local game adapter for training an AI to drive in
[PolyTrack](https://www.kodub.com/apps/polytrack).

The project is deliberately split into two parts:

- a Python Gymnasium environment, trainer, evaluation tools, and deterministic mock simulator;
- a PolyModLoader mod that manually steps PolyTrack's simulation worker and exposes telemetry over
  a localhost-only WebSocket connection.

The mock backend is usable before any game files are present. It validates the training loop,
reward calculation, action encoding, and baseline controller without coupling those components to
minified game internals.

The deterministic mock remains useful for testing policy code without starting the game. The real
adapter targets PolyTrack 0.6.3 (with 0.6.2 compatibility) through PolyModLoader; leaderboard
submissions and multiplayer are disabled while the mod is loaded.

## Quick start

Python 3.11 or newer is required.

```text
python -m venv .venv
# Activate first: Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev,train,gui]"
python -m pytest
polybot-gui
```

Activate the environment using the normal command for your shell, or invoke its Python directly.
The graphical application is the primary training interface on Windows and Linux. It supports
PPO and TQC, algorithm-scoped models, resume/stop, CPU/CUDA selection, curriculum, and rolling
checkpoints. See [`docs/training.md`](docs/training.md).

Train a first PPO policy in the mock environment:

```powershell
polybot-train --algorithm ppo --backend mock --timesteps 100000
```

Train a TQC policy with continuous steering and signed throttle/brake:

```powershell
polybot-train --algorithm tqc --backend mock --track mock/gentle-s --timesteps 100000
```

PPO remains the default for older `polybot-train` commands. PPO selects a MultiDiscrete action
with 41-level PWM steering and learns on-policy. TQC selects a two-value continuous action,
converts it to digital controls at 1 ms physics ticks, and learns off-policy from a replay buffer.
TQC is an alternative for controlled benchmarking, not an established improvement over PPO.

Evaluate a saved policy deterministically:

```powershell
polybot-eval models/mock-gentle-s/tqc/latest.zip --backend mock --episodes 5
```

Evaluation and playback read the algorithm from model metadata. For older archives without
metadata, pass `--algorithm ppo` (or `tqc`) explicitly. New GUI models live in
`models/<track>/ppo/` and `models/<track>/tqc/`; existing top-level PPO archives remain where they
are and can still be selected.

## Drive the real game

Open [PolyModLoader](https://web.polymodloader.com/) and add this mod URL once:

```text
https://cdn.polymodloader.com/gh/michael201110/poly-bot/main/pml-mod
```

Choose `latest`, click **Load**, then **Apply**. In a terminal, install the command and start it:

```powershell
python -m pip install -e ".[dev,train]"
polybot-drive --centerline
```

Then choose a track in PolyTrack, load a ghost lap, and enter its race. If the race was already
open, restart it after enabling the mod.
The adapter uses that ghost as its route reference. The built-in centreline controller is a
wiring test; to drive with a trained PPO policy instead, use:

```powershell
polybot-drive --model models/polybot-ppo
```

`polybot-drive` defaults to `--track current --frame-skip 10` and listens only on
`ws://127.0.0.1:8765`. The mod permits read-only ghost downloads but blocks record submissions and
multiplayer connections.

To train against the real worker instead of driving one episode:

```powershell
polybot-train --backend websocket --timesteps 1000000 --model-out models/polybot-real
```

WebSocket training also selects `current` and a 10-tick action repeat automatically. See
[`docs/game-integration.md`](docs/game-integration.md) for the integration seam and
[`docs/protocol.md`](docs/protocol.md) for the wire format.

## Maintenance checks

```text
python -m pip install -e ".[dev,train,gui]"
python -m pytest
python -m ruff check .
python tools/validate_pml_mod.py
```

The manifest check covers both supported game versions; it does not launch the game or verify
bundles unless `--worker` and `--main` are supplied. See the
[bundle validation instructions](docs/game-integration.md#bundle-validation) for pinned 0.6.3
checks and the manual in-game smoke test. The Python package version (`0.1.0`), mod release
(`0.1.29`), game version (`0.6.3`), and wire protocol (`2`) are independent.

## Design principles

- **Local automation only.** The agent must not submit leaderboard records or automate PolyTrack
  servers.
- **Simulation first.** Training uses telemetry and fixed physics steps; pixels can be added later.
- **Versioned boundary.** Game internals are isolated behind a narrow adapter so PolyTrack updates
  do not require rewriting the trainer.
- **Test before optimize.** A deterministic mock and a hand-written controller verify the full
  environment before reinforcement learning is introduced.

## Repository layout

```text
bridge/                 Legacy protocol-v1 bridge and game-adapter template
docs/                   Protocol and integration notes
pml-mod/                PolyModLoader package for the real 0.6.2/0.6.3 simulation worker
src/polybot/            Environment, transport, training services, GUI, controller, CLI
tests/                  Protocol, determinism, reward, and controller tests
```

## Safety and fair play

This project is intended for local research and clearly labelled AI demonstrations. The mod blocks
the game's write and multiplayer entry points and allows local finish feedback before restarting; still review
PolyTrack's current terms before distributing a modified build.

## Contributor Hall of Fame

- **[Gotchaaaaaa](https://github.com/Gotchaaaaaa)** — first external contributor; added PolyTrack 0.6.3 support.
