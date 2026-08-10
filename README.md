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
adapter targets PolyTrack 0.6.2 through PolyModLoader; leaderboard submissions and multiplayer are
disabled while the mod is loaded.

## Quick start

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,train]"
pytest
polybot-smoke
```

Train a first PPO policy in the mock environment:

```powershell
polybot-train --backend mock --timesteps 100000
```

Evaluate a saved policy deterministically:

```powershell
polybot-eval models/polybot-ppo --backend mock --episodes 5
```

## Drive the real game

Open [PolyModLoader](https://web.polymodloader.com/) and add this mod URL once:

```text
https://cdn.polymodloader.com/gh/michael201110/poly-bot/main/pml-mod
```

Choose `latest`, click **Load**, then **Apply**. In PowerShell, install the command and start it:

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
bridge/                 JavaScript bridge and game-adapter template
docs/                   Protocol and integration notes
pml-mod/                PolyModLoader package for the real 0.6.2 simulation worker
src/polybot/            Gymnasium environment, transport, mock, controller, CLI
tests/                  Protocol, determinism, reward, and controller tests
```

## Safety and fair play

This project is intended for local research and clearly labelled AI demonstrations. The mod blocks
the game's write and multiplayer entry points and hides AI finish state from the UI; still review
PolyTrack's current terms before distributing a modified build.
