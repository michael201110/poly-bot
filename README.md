# poly-bot

An offline reinforcement-learning environment for training an AI to drive in
[PolyTrack](https://www.kodub.com/apps/polytrack).

The project is deliberately split into two parts:

- a Python Gymnasium environment, trainer, evaluation tools, and deterministic mock simulator;
- a small JavaScript bridge that adapts PolyTrack's simulation worker to the versioned protocol.

The mock backend is usable before any game files are present. It validates the training loop,
reward calculation, action encoding, and baseline controller without coupling those components to
minified game internals.

## Status

The training foundation and bridge contract are implemented. The remaining game-specific task is
to implement the four methods in `bridge/polytrack_game_api.stub.js` against a local PolyTrack
desktop build. No leaderboard or multiplayer integration is included.

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

To connect a local game adapter, start the trainer in WebSocket mode and then load the bridge in
the game:

```powershell
polybot-train --backend websocket --host 127.0.0.1 --port 8765
```

The Python process listens only on localhost by default. See
[`docs/game-integration.md`](docs/game-integration.md) for the integration seam and
[`docs/protocol.md`](docs/protocol.md) for the wire format.

## Design principles

- **Offline only.** The agent must not submit leaderboard records or automate PolyTrack servers.
- **Simulation first.** Training uses telemetry and fixed physics steps; pixels can be added later.
- **Versioned boundary.** Game internals are isolated behind a narrow adapter so PolyTrack updates
  do not require rewriting the trainer.
- **Test before optimize.** A deterministic mock and a hand-written controller verify the full
  environment before reinforcement learning is introduced.

## Repository layout

```text
bridge/                 JavaScript bridge and game-adapter template
docs/                   Protocol and integration notes
src/polybot/            Gymnasium environment, transport, mock, controller, CLI
tests/                  Protocol, determinism, reward, and controller tests
```

## Safety and fair play

This project is intended for local research and clearly labelled AI demonstrations. Keep the game
offline while training, do not submit automated runs to public leaderboards, and review PolyTrack's
current terms before distributing a modified build.
