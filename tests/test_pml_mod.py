from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
MOD_ROOT = REPOSITORY / "pml-mod"


def _load_validator():
    path = REPOSITORY / "tools" / "validate_pml_mod.py"
    spec = importlib.util.spec_from_file_location("validate_pml_mod", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("game_version", ["0.6.2", "0.6.3"])
def test_pml_manifest_resolves_versioned_entry_point(game_version: str) -> None:
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    version = manifest["latest"][game_version]
    version_manifest = json.loads((MOD_ROOT / version / "version.json").read_text(encoding="utf-8"))

    assert manifest["id"] == "polybot-bridge"
    assert version_manifest == {
        "targets": [game_version],
        "dependencies": [],
        "main": "main.mod.js",
    }
    assert (MOD_ROOT / version / version_manifest["main"]).is_file()
    runtime = (MOD_ROOT / version / "worker_runtime.js").read_text(encoding="utf-8")
    main_source = (MOD_ROOT / version / version_manifest["main"]).read_text(encoding="utf-8")
    assert 'from "./worker_runtime.js"' not in main_source
    assert (
        runtime.replace("export function polybotWorkerInjection()", "function polybotWorkerInjection()", 1)
        in main_source
    )
    if version == "0.1.42":
        renderer = (MOD_ROOT / version / "hud_renderer.mjs").read_text(encoding="utf-8")
        assert renderer.replace(
            "export function installPolyBotHudOverlay(",
            "function installPolyBotHudOverlay(",
            1,
        ) in main_source


def test_worker_and_offline_anchors_are_declared_once_in_mod_source() -> None:
    validator = _load_validator()
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    source = (MOD_ROOT / manifest["latest"]["0.6.3"] / "main.mod.js").read_text(encoding="utf-8")

    for token in (*validator.WORKER_TOKENS, *validator.MAIN_TOKENS):
        assert token in source
    assert 'ta, "m", Ps' in source
    assert "__polybotBindVisualReplayRenderer" in source


def test_latest_063_bridge_adds_hud_without_changing_older_releases() -> None:
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["latest"]["0.6.3"] == "0.1.42"
    latest_worker = (MOD_ROOT / "0.1.42" / "worker_runtime.js").read_text(encoding="utf-8")
    old_worker = (MOD_ROOT / "0.1.38" / "worker_runtime.js").read_text(encoding="utf-8")
    assert '"ai_overlay_hud"' in latest_worker
    assert 'request.op === "hud_frame"' in latest_worker
    assert '"ai_overlay_hud"' not in old_worker


def test_worker_connects_when_player_is_created_and_started() -> None:
    source = (MOD_ROOT / "0.1.0" / "worker_runtime.js").read_text(encoding="utf-8")
    create_case = source.split("case messageTypes.CreateCar:", 1)[1].split("case messageTypes.DeleteCar:", 1)[0]
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
    assert "replayTicks % 250 === 0" not in source
    assert "fast-forwards to a timed curriculum start" not in source
    assert "params.native_restart !== false" in source
    assert "activeRequestToken = null" in source
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
    assert "setTimeout(pauseManualCars, finishDisplayDelayMs + 100);" in source
    assert "setTimeout(resolve, finishDisplayDelayMs + 100)" not in source
    assert "polybotPlayerFinished: playerFinished" in source
    assert "visibleBytes[11] &= ~2" not in source


def test_latest_mod_uses_native_backspace_after_finish() -> None:
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    source = (MOD_ROOT / manifest["latest"]["0.6.3"] / "main.mod.js").read_text(encoding="utf-8")

    assert "__polybotWrapSimulationWorker" in source
    assert "polybotPlayerFinished" in source
    assert 'new KeyboardEvent("keydown", eventOptions)' in source
    assert 'new KeyboardEvent("keyup", eventOptions)' in source
    assert 'code: "Backspace"' in source


def test_latest_mod_bundles_worker_without_a_stale_import() -> None:
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    source = (MOD_ROOT / manifest["latest"]["0.6.3"] / "main.mod.js").read_text(encoding="utf-8")

    assert "function polybotWorkerInjection()" in source
    assert "import { polybotWorkerInjection }" not in source
    assert "/v0.1.23/pml-mod/0.1.0/worker_runtime.js" not in source


def test_latest_worker_keeps_curriculum_reset_kinematics_and_action_history() -> None:
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    version = manifest["latest"]["0.6.3"]
    source = (MOD_ROOT / version / "worker_runtime.js").read_text(encoding="utf-8")
    assert "function velocityBetween(previous, current, dtSeconds, basisOverride = null)" in source
    assert "previousPlayerBuffer" in source
    assert "earlierPlayerBuffer" in source
    assert "previousAction: initialAction" in source
    assert "kinematicDtSeconds ?? ticksAdvanced * fixedDtSeconds" in source
    assert "null,\n              previousPlayerBuffer ? fixedDtSeconds : 0" in source


@pytest.mark.parametrize("version", ["0.1.35", "0.1.36"])
def test_stage_four_replay_commands_reuse_bridge_and_keep_training_dispatch(
    version: str,
) -> None:
    source = (MOD_ROOT / version / "worker_runtime.js").read_text(encoding="utf-8")
    for operation in (
        "visual_replay_begin",
        "visual_replay_chunk",
        "visual_replay_commit",
        "visual_replay_play",
        "visual_replay_pause",
        "visual_replay_restart",
        "visual_replay_seek",
        "visual_replay_speed",
        "visual_replay_opacity",
        "visual_replay_color",
        "visual_replay_end_behavior",
        "visual_replay_fade_duration",
        "visual_replay_clear",
    ):
        assert f'case "{operation}"' in source
    for operation in ("hello", "reset", "step", "close"):
        assert f'case "{operation}"' in source


@pytest.mark.parametrize(
    ("version", "constructor"),
    [
        ("0.1.35", "new U.A("),
        ("0.1.36", "new z.A("),
    ],
)
def test_stage_four_ghost_uses_native_renderer_car_without_physics_manager(
    version: str,
    constructor: str,
) -> None:
    source = (MOD_ROOT / version / "main.template.js").read_text(encoding="utf-8")
    create_car = source.split("createCar: (state) =>", maxsplit=1)[1].split("\n        ),", maxsplit=1)[0]

    assert constructor in create_car
    assert "null, state, null, null," in create_car
    assert "CreateCar" not in create_car


@pytest.mark.parametrize(
    ("version", "constructor", "hook"),
    [
        ("0.1.37", "new U.A(", 'update(e) {\n              const t = (0, R.gn)(this, jr, "m", ys).call(this);'),
        ("0.1.42", "new z.A(", 'update(e) {\n              const t = (0, R.gn)(this, ta, "m", Ps).call(this);'),
    ],
)
def test_current_swarm_releases_use_native_render_only_cars_and_exact_update_hooks(
    version: str,
    constructor: str,
    hook: str,
) -> None:
    template = (MOD_ROOT / version / "main.template.js").read_text(encoding="utf-8")
    worker = (MOD_ROOT / version / "worker_runtime.js").read_text(encoding="utf-8")
    bundled = (MOD_ROOT / version / "main.mod.js").read_text(encoding="utf-8")

    assert template.count(hook.replace("\n", "\\n")) == 1
    assert constructor in template
    renderer = template.split("createCar: (state) =>", maxsplit=1)[1].split("\n        ),", maxsplit=1)[0]
    assert "null, state, null, null," in renderer
    assert "CreateCar" not in renderer
    for operation in (
        "visual_replay_swarm_begin",
        "visual_replay_swarm_episode_begin",
        "visual_replay_swarm_chunk",
        "visual_replay_swarm_episode_commit",
        "visual_replay_swarm_commit",
        "visual_replay_status",
    ):
        assert f'case "{operation}"' in worker
        assert operation in bundled


def test_v2_worker_exports_vehicle_dynamics_and_ghost_guidance() -> None:
    source = (MOD_ROOT / "0.1.27" / "worker_runtime.js").read_text(encoding="utf-8")
    for field in (
        "local_acceleration_mps2",
        "suspension_lengths_m",
        "suspension_velocities_mps",
        "wheel_skids",
        "actual_steering",
        "ghost_relative_position_m",
        "ghost_heading_error_rad",
        "ghost_target_speed_mps",
        "expert_action",
    ):
        assert field in source


def test_latest_ghost_guidance_is_position_aligned() -> None:
    source = (MOD_ROOT / "0.1.28" / "worker_runtime.js").read_text(encoding="utf-8")
    assert "const timedGuide = guide;" in source
    assert "const timedIndex = referenceIndex;" in source
    assert "const timedGuide = reference.points[session.timedReferenceIndex];" not in source


def test_latest_mod_uses_native_backspace_before_aborted_reset() -> None:
    worker_source = (MOD_ROOT / "0.1.0" / "worker_runtime.js").read_text(encoding="utf-8")
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    main_source = (MOD_ROOT / manifest["latest"]["0.6.3"] / "main.mod.js").read_text(encoding="utf-8")

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
    source = (MOD_ROOT / manifest["latest"]["0.6.3"] / "main.mod.js").read_text(encoding="utf-8")

    assert 'if (rewind && (0, l.gn)(this, ie, "f"))' in source
    assert '(0, l.GG)(this, re, new st.A(), "f");' in source


def test_latest_mod_treats_equal_record_frames_as_rewinds() -> None:
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    source = (MOD_ROOT / manifest["latest"]["0.6.3"] / "main.mod.js").read_text(encoding="utf-8")

    assert "e.frames <= previous.frames" in source


def test_anchor_validator_rejects_missing_or_duplicate_tokens(tmp_path: Path) -> None:
    validator = _load_validator()
    worker = tmp_path / "worker.js"
    main = tmp_path / "main.js"
    worker.write_text(validator.WORKER_TOKENS[0] * 2, encoding="utf-8")
    main.write_text("\n".join(validator.MAIN_TOKENS[:-1]), encoding="utf-8")

    failures = validator.validate(worker, main, require_pinned_hash=False)

    assert any("found 2" in failure for failure in failures)
    assert any("found 0" in failure for failure in failures)


@pytest.mark.parametrize("game_version", ["0.6.2", "0.6.3"])
def test_validator_checks_raw_bundles_and_version_specific_hashes(
    tmp_path: Path,
    game_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator()
    worker = tmp_path / "worker.js"
    main = tmp_path / "main.js"
    worker.write_text("\n".join(validator.WORKER_TOKENS), encoding="utf-8")
    main.write_text(
        "\n".join(
            (
                *validator.MAIN_TOKENS,
                *validator.REPLAY_RENDER_TOKENS[game_version],
            )
        ).replace(validator.PML_WORKER_CONSTRUCTOR, validator.RAW_WORKER_CONSTRUCTOR),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        validator.PINNED_HASHES,
        game_version,
        (validator._sha256(worker), validator._sha256(main)),
    )
    assert validator.validate(worker, main, game_version=game_version) == []
    worker.write_text(worker.read_text() + "\n// changed bundle", encoding="utf-8")
    failures = validator.validate(worker, main, game_version=game_version)
    assert len(failures) == 1
    assert f"pinned {game_version} worker hash" in failures[0]
    assert validator.validate(worker, main, game_version=game_version, require_pinned_hash=False) == []


def test_manifests_cover_all_supported_games() -> None:
    assert _load_validator()._validate_manifests(REPOSITORY) == []


def test_manifest_validator_rejects_missing_target_and_entry_point(tmp_path: Path) -> None:
    root = tmp_path / "pml-mod"
    release = root / "test"
    release.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"latest": {"0.6.2": "test", "0.6.3": "test"}}), encoding="utf-8")
    (release / "version.json").write_text(json.dumps({"targets": ["0.6.2"], "main": "missing.js"}), encoding="utf-8")
    failures = _load_validator()._validate_manifests(tmp_path)
    assert any("does not target PolyTrack 0.6.3" in failure for failure in failures)
    assert any("missing mod entry point" in failure for failure in failures)
