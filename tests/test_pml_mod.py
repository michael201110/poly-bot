from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
MOD_ROOT = REPOSITORY / "pml-mod"


def _load_validator():
    path = REPOSITORY / "tools" / "validate_pml_mod.py"
    spec = importlib.util.spec_from_file_location("validate_pml_mod", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pml_manifest_resolves_versioned_entry_point() -> None:
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    version = manifest["latest"]["0.6.2"]
    version_manifest = json.loads((MOD_ROOT / version / "version.json").read_text(encoding="utf-8"))

    assert manifest["id"] == "polybot-bridge"
    assert version_manifest == {
        "targets": ["0.6.2"],
        "dependencies": [],
        "main": "main.mod.js",
    }
    assert (MOD_ROOT / version / version_manifest["main"]).is_file()
    runtime_version = "0.1.0"
    assert (MOD_ROOT / runtime_version / "worker_runtime.js").is_file()

    main_source = (MOD_ROOT / version / version_manifest["main"]).read_text(encoding="utf-8")
    assert 'from "./worker_runtime.js"' not in main_source
    expected = (
        "https://cdn.polymodloader.com/gh/michael201110/"
        f"poly-bot/v{version}/pml-mod/{runtime_version}/worker_runtime.js"
    )
    assert f'from "{expected}"' in main_source


def test_worker_and_offline_anchors_are_declared_once_in_mod_source() -> None:
    validator = _load_validator()
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    source = (MOD_ROOT / manifest["latest"]["0.6.2"] / "main.mod.js").read_text(encoding="utf-8")

    for token in (*validator.WORKER_TOKENS, *validator.MAIN_TOKENS):
        assert token in source


def test_worker_connects_when_player_is_created_and_started() -> None:
    source = (MOD_ROOT / "0.1.0" / "worker_runtime.js").read_text(encoding="utf-8")
    create_case = source.split("case messageTypes.CreateCar:", 1)[1].split(
        "case messageTypes.DeleteCar:", 1
    )[0]
    start_case = source.split("case messageTypes.StartCar:", 1)[1].split("default:", 1)[0]

    assert "connectSocket();" in create_case
    assert "message.carId === playerCarId" in start_case
    assert "connectSocket();" in start_case


def test_worker_can_start_a_stationary_player_car() -> None:
    source = (MOD_ROOT / "0.1.0" / "worker_runtime.js").read_text(encoding="utf-8")

    assert "return Boolean(playerMessage && chooseGhostMessage(playerMessage));" in source
    assert "if (!startMessages.has(playerMessage.carId))" in source
    assert "targetSimulationTimeFrames: null" in source


def test_ghost_pose_reward_uses_the_same_elapsed_reference_frame() -> None:
    source = (MOD_ROOT / "0.1.0" / "worker_runtime.js").read_text(encoding="utf-8")

    assert "function findTimedReferencePoint()" in source
    assert "session.startReferenceFrame + session.tick" in source
    assert "distance(decoded.position, timedGuide.position)" in source
    assert "normalizeQuaternion(timedGuide.quaternion)" in source
    assert "point.frame * fixedDtSeconds >= startTimeSeconds" in source
    assert "const timedGuide = reference.points[session.timedReferenceIndex];" in source
    assert "replayTicks % 250 === 0" in source
    assert "setTimeout(resolve, 0)" in source
    assert "params.native_restart !== false" in source
    assert "activeRequestToken = null" in source
    assert "trainer disconnected during timed reset" in source
    assert "activeRequestToken?.sourceSocket === nextSocket" in source


def test_worker_preserves_transient_collision_impulses_across_frame_skip() -> None:
    source = (MOD_ROOT / "0.1.0" / "worker_runtime.js").read_text(encoding="utf-8")

    assert "collisionImpulsePeak = Math.max" in source
    assert "[finite(collisionImpulsePeak)]" in source


def test_worker_publishes_checkpoint_transitions_and_local_finish_state() -> None:
    source = (MOD_ROOT / "0.1.0" / "worker_runtime.js").read_text(encoding="utf-8")

    assert "decoded.nextCheckpointIndex > visibleCheckpoint" in source
    assert "publishStates([buffer]);" in source
    assert '"local_finish_ui"' in source


def test_worker_coasts_in_realtime_before_resetting_a_finish() -> None:
    source = (MOD_ROOT / "0.1.0" / "worker_runtime.js").read_text(encoding="utf-8")

    assert "const finishDisplayDelayMs = 500;" in source
    assert "async function stepEpisode(params)" in source
    assert "car.isPaused = false;" in source
    assert "setTimeout(resolve, finishDisplayDelayMs + 100)" in source
    assert "pauseManualCars();" in source
    assert "polybotPlayerFinished: playerFinished" in source
    assert "visibleBytes[11] &= ~2" not in source


def test_latest_mod_uses_native_backspace_after_finish() -> None:
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    source = (MOD_ROOT / manifest["latest"]["0.6.2"] / "main.mod.js").read_text(encoding="utf-8")

    assert "__polybotWrapSimulationWorker" in source
    assert "polybotPlayerFinished" in source
    assert 'new KeyboardEvent("keydown", eventOptions)' in source
    assert 'new KeyboardEvent("keyup", eventOptions)' in source
    assert 'code: "Backspace"' in source


def test_latest_mod_uses_native_backspace_before_aborted_reset() -> None:
    worker_source = (MOD_ROOT / "0.1.0" / "worker_runtime.js").read_text(encoding="utf-8")
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    main_source = (MOD_ROOT / manifest["latest"]["0.6.2"] / "main.mod.js").read_text(
        encoding="utf-8"
    )

    assert "polybotAbortRestart: true" in worker_source
    assert "aborted run does not need the finish-only display pause." in worker_source
    assert "await new Promise((resolve) => setTimeout(resolve, 0));" in worker_source
    assert "event.data?.polybotAbortRestart !== true" in main_source
    assert "const restartDelayMs = playerFinished ? 500 : 0;" in main_source
    assert "}, restartDelayMs);" in main_source
    assert "pressBackspace();" in main_source
    assert "restartScheduled" in main_source


def test_latest_mod_resets_the_main_thread_control_recorder() -> None:
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    source = (MOD_ROOT / manifest["latest"]["0.6.2"] / "main.mod.js").read_text(encoding="utf-8")

    assert '(0, l.GG)(this, Ue, null, "f"),' in source
    assert '(0, l.GG)(this, re, new st.A(), "f"),' in source


def test_anchor_validator_rejects_missing_or_duplicate_tokens(tmp_path: Path) -> None:
    validator = _load_validator()
    worker = tmp_path / "worker.js"
    main = tmp_path / "main.js"
    worker.write_text(validator.WORKER_TOKENS[0] * 2, encoding="utf-8")
    main.write_text("\n".join(validator.MAIN_TOKENS[:-1]), encoding="utf-8")

    failures = validator.validate(worker, main, require_pinned_hash=False)

    assert any("found 2" in failure for failure in failures)
    assert any("found 0" in failure for failure in failures)
