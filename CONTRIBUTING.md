# Contributing to PolyBot

Thanks for helping improve PolyBot. Changes to the simulator protocol, observations, actions, rewards, model metadata, and saved checkpoints can affect whether existing experiments remain usable, so please describe any behavior or data-format changes clearly.

## Set up a development environment

PolyBot requires Python 3.11 or later. From the repository root, create an environment and install the development, training, and GUI dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,train,gui]"
```

For changes that do not need the training or GUI extras, `python -m pip install -e ".[dev]"` is sufficient. Run checks before opening a pull request:

```powershell
python -m pytest
python -m ruff check .
python tools/validate_pml_mod.py
```

The PML validation checks the game adapter. Training changes should also be smoke-tested against the mock backend when practical; do not require a live game connection for routine CI tests.

## Make a change

- Keep changes focused and explain user-visible behavior in the description.
- Add or update tests for behavior changes, especially for reward calculations, control mapping, checkpoint continuation, and protocol compatibility.
- Preserve backward compatibility for saved models and configuration where practical. If compatibility changes, document the migration and affected versions.
- Do not commit generated training runs, logs, caches, credentials, or model artifacts unless the change explicitly requires a reviewed fixture.
- Use Ruff formatting conventions and keep public CLI/GUI options documented in the README or relevant guide.

## Pull requests

Open a pull request with a concise description of the problem and solution. Include the checks you ran, note any checks you could not run, and attach screenshots for meaningful GUI changes. Link related issues and call out changes to model formats, reward semantics, game versions, or training defaults. A maintainer will review the behavior and validation before merging.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
