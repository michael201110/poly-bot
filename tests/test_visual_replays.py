from __future__ import annotations

import json
import logging

import gymnasium as gym
import numpy as np
import pytest

from polybot.mock import MockSimulatorTransport
from polybot.training.config import TQCConfig, TrainingConfig
from polybot.training.runner import TrainingRunner
from polybot.training.visual_replays import (
    INDEX_SCHEMA,
    AsyncReplayWriter,
    ReplayFormatError,
    ReplayPayload,
    ReplaySample,
    VisualReplayCaptureWrapper,
    VisualReplaySession,
    load_replay_episode,
    load_replay_index,
)


class MemoryWriter:
    def __init__(self) -> None:
        self.payloads: list[ReplayPayload] = []

    def submit(self, payload: ReplayPayload) -> bool:
        self.payloads.append(payload)
        return True


def telemetry(*, tick: int, elapsed_s: float, x: float = 0.0) -> dict:
    return {
        "tick": tick,
        "elapsed_s": elapsed_s,
        "position_m": [x, 1.0, 2.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "route_progress_m": x,
        "track_length_m": 100.0,
        "events": (),
        "simulator_info": {"simulator": "polytrack-local", "game_version": "0.6.3"},
    }


def make_session(
    writer: MemoryWriter,
    *,
    step: int = 37,
    sample_hz: float = 20.0,
    record_observations: bool = False,
) -> VisualReplaySession:
    return VisualReplaySession(
        writer, run_id="run-1", algorithm="tqc", track_id="official/01",
        track_name="Summer 1", frame_skip=30, sample_hz=sample_hz,
        record_observations=record_observations, training_step_provider=lambda: step,
    )


def test_transform_sample_roundtrip_preserves_arbitrary_timestamps(tmp_path) -> None:
    writer = AsyncReplayWriter(tmp_path, run_metadata={"run_id": "run-1"})
    samples = [
        ReplaySample(5, 0.005, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)),
        ReplaySample(6, 0.006, (1.1, 2.0, 3.0), (0.0, 0.1, 0.0, 0.995)),
    ]
    metadata = {
        "schema": "polybot.visual-replay.v1", "run_id": "run-1",
        "episode_id": "episode-000001", "algorithm": "tqc",
        "track_id": "official/01", "track_name": "Summer 1",
        "training_step": 10, "training_step_start": 10, "training_step_end": 11,
        "episode_length_decisions": 1, "episode_length_ticks": 1,
        "status": "finished", "final_progress_m": 100.0, "frame_skip": 30,
        "sample_count": 2, "file": "episode-000001.npz",
    }
    assert writer.submit(ReplayPayload(metadata, samples))
    writer.close()

    entry = load_replay_index(tmp_path)[0]
    loaded = load_replay_episode(tmp_path, entry)
    assert [sample.tick for sample in loaded.samples] == [sample.tick for sample in samples]
    assert [sample.elapsed_s for sample in loaded.samples] == [sample.elapsed_s for sample in samples]
    np.testing.assert_allclose(
        [sample.position_m for sample in loaded.samples],
        [sample.position_m for sample in samples],
    )
    np.testing.assert_allclose(
        [sample.quaternion_xyzw for sample in loaded.samples],
        [sample.quaternion_xyzw for sample in samples],
    )
    assert loaded.metadata["training_step"] == 10


def test_optional_policy_arrays_roundtrip_in_distinct_npz_arrays(tmp_path) -> None:
    writer = AsyncReplayWriter(tmp_path, run_metadata={"run_id": "run-1"})
    payload = ReplayPayload(
        metadata={
            "schema": "polybot.visual-replay.v1", "run_id": "run-1",
            "episode_id": "episode-000001", "algorithm": "ppo",
            "track_id": "track", "track_name": "Track",
            "training_step": 0, "training_step_start": 0, "training_step_end": 1,
            "episode_length_decisions": 1, "episode_length_ticks": 30,
            "status": "failed", "final_progress_m": 2.0, "frame_skip": 30,
            "sample_count": 1, "file": "episode-000001.npz",
        },
        samples=[ReplaySample(30, 0.03, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))],
        decision_ticks=[30],
        decision_elapsed_s=[0.03],
        observations=[np.array([1.0, 2.0], dtype=np.float32)],
        actions=[np.array([0.5], dtype=np.float32)],
    )
    writer.submit(payload)
    writer.close()
    [entry] = load_replay_index(tmp_path)
    loaded = load_replay_episode(tmp_path, entry)
    assert loaded.decision_ticks == [30]
    assert loaded.decision_elapsed_s == [0.03]
    np.testing.assert_array_equal(loaded.observations, [[1.0, 2.0]])
    np.testing.assert_array_equal(loaded.actions, [[0.5]])


def test_index_metadata_and_backward_compatibility_without_replays(tmp_path) -> None:
    assert load_replay_index(tmp_path / "old-run") == []

    writer = AsyncReplayWriter(tmp_path / "visual_replays" / "run-1", run_metadata={"run_id": "run-1"})
    session = make_session(MemoryWriter())
    session.writer = writer
    session.reset(telemetry(tick=200, elapsed_s=0.0))
    final_info = telemetry(tick=230, elapsed_s=0.03, x=3.0)
    final_info["events"] = ("finish",)
    session.step(np.array([0.1]), np.array([0.2]), final_info,
                 terminated=True, truncated=False)
    writer.close()

    index_path = tmp_path / "visual_replays" / "run-1" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["schema"] == INDEX_SCHEMA
    [entry] = load_replay_index(index_path.parent)
    assert entry["training_step"] == entry["training_step_start"] == 37
    assert entry["training_step_end"] == 38
    assert entry["episode_length_decisions"] == 1
    assert entry["episode_length_ticks"] == 30
    assert entry["frame_skip"] == 30
    assert entry["game_version"] == "0.6.3"
    assert entry["status"] == "finished"


def test_sampling_caps_visual_rate_and_keeps_episode_end() -> None:
    writer = MemoryWriter()
    session = make_session(writer, step=100, sample_hz=20.0)
    session.reset(telemetry(tick=0, elapsed_s=0.0))
    for sample_index in range(1, 7):
        session.step(
            np.array([0.0]), np.array([0.0]),
            telemetry(tick=sample_index, elapsed_s=sample_index * 0.01, x=float(sample_index)),
            terminated=sample_index == 6, truncated=False,
        )
    [payload] = writer.payloads
    assert [sample.tick for sample in payload.samples] == [0, 5, 6]
    assert payload.metadata["sample_hz_limit"] == 20.0


def test_observations_and_actions_are_separate_and_opt_in() -> None:
    writer = MemoryWriter()
    session = make_session(writer, record_observations=True)
    session.reset(telemetry(tick=10, elapsed_s=0.0))
    session.step(
        np.array([1.0, 2.0]), np.array([0.25, -0.5]),
        telemetry(tick=40, elapsed_s=0.03), terminated=False, truncated=True,
    )
    [payload] = writer.payloads
    assert payload.metadata["observations_recorded"] is True
    assert payload.observations is not None
    assert payload.actions is not None
    np.testing.assert_array_equal(payload.observations[0], [1.0, 2.0])
    np.testing.assert_array_equal(payload.actions[0], [0.25, -0.5])
    assert payload.decision_ticks == [40]
    assert payload.metadata["status"] == "truncated"


def test_capture_wrapper_preserves_environment_outputs_and_finalizes_partial_episode() -> None:
    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(-10, 10, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.array([0.0], dtype=np.float32), telemetry(tick=90, elapsed_s=0.0)

        def step(self, action):
            return (
                np.array([1.0], dtype=np.float32), 2.0, False, False,
                telemetry(tick=120, elapsed_s=0.03, x=4.0),
            )

    writer = MemoryWriter()
    wrapper = VisualReplayCaptureWrapper(TinyEnv(), make_session(writer))
    obs, reset_info = wrapper.reset()
    result = wrapper.step(np.array([0.5], dtype=np.float32))
    assert obs.tolist() == [0.0]
    assert reset_info["tick"] == 90
    assert result[0].tolist() == [1.0]
    assert result[1:4] == (2.0, False, False)
    wrapper.close()
    [payload] = writer.payloads
    assert payload.metadata["status"] == "interrupted"
    assert payload.metadata["training_step_start"] == 37
    assert payload.metadata["training_step_end"] == 38


def test_empty_autoreset_episode_is_not_saved_as_a_training_attempt() -> None:
    writer = MemoryWriter()
    session = make_session(writer)
    session.reset(telemetry(tick=10, elapsed_s=0.0))
    session.close()
    assert writer.payloads == []


def test_training_runner_wraps_only_the_training_environment_for_capture(tmp_path) -> None:
    config = TrainingConfig(
        algorithm="tqc", backend="websocket", track_id="mock/straight",
        track_name="Mock straight", device="cpu", output_root=tmp_path / "models",
        log_root=tmp_path / "logs", tqc=TQCConfig(architecture="tiny"),
    )
    runner = TrainingRunner(config, transport_factory=MockSimulatorTransport)
    runner.model = type("Model", (), {"num_timesteps": 19})()
    runner._start_visual_replay_session("run-1")
    env = runner._environment(record_visual_replays=True)
    try:
        observation, _ = env.reset(seed=3)
        result = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert result[0].shape == observation.shape
    finally:
        env.close()
    assert runner._visual_replay_session is not None
    runner._visual_replay_session.shutdown()
    directory = runner.registry.algorithm_dir("Mock straight", "tqc") / "visual_replays" / "run-1"
    [entry] = load_replay_index(directory)
    assert entry["training_step_start"] == 19
    assert entry["training_step_end"] == 20
    assert entry["episode_length_decisions"] == 1


@pytest.mark.parametrize("bad_file", ["missing.npz", "corrupt.npz", "incomplete.npz"])
def test_corrupt_or_incomplete_indexed_payloads_are_reported(tmp_path, bad_file) -> None:
    entry = {
        "run_id": "run-1", "episode_id": "episode-000001", "algorithm": "tqc",
        "track_id": "track", "track_name": "Track", "training_step": 0,
        "training_step_start": 0, "training_step_end": 1, "episode_length_decisions": 1,
        "episode_length_ticks": 30, "status": "failed", "final_progress_m": 1.0,
        "frame_skip": 30, "sample_count": 1, "file": bad_file,
    }
    if bad_file == "corrupt.npz":
        (tmp_path / bad_file).write_bytes(b"not an npz archive")
    elif bad_file == "incomplete.npz":
        np.savez_compressed(tmp_path / bad_file, ticks=np.array([1]))
    with pytest.raises(ReplayFormatError):
        load_replay_episode(tmp_path, entry)


def test_incomplete_index_entry_and_invalid_metadata_are_rejected(tmp_path) -> None:
    (tmp_path / "index.json").write_text(
        json.dumps({"schema": INDEX_SCHEMA, "episodes": [{"episode_id": "missing-fields"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ReplayFormatError, match="metadata is missing"):
        load_replay_index(tmp_path)


def test_async_queue_is_bounded_and_failed_writes_warn_without_raising(tmp_path, caplog) -> None:
    writer = AsyncReplayWriter(tmp_path / "not-a-directory", run_metadata={"run_id": "run-1"})
    writer.directory.mkdir(parents=True)
    (writer.directory / "episode-000001.npz").mkdir()
    payload = ReplayPayload(
        metadata={
            "run_id": "run-1", "episode_id": "episode-000001", "algorithm": "tqc",
            "track_id": "track", "track_name": "Track", "training_step": 0,
            "training_step_start": 0, "training_step_end": 0,
            "episode_length_decisions": 0, "episode_length_ticks": 0,
            "status": "interrupted", "final_progress_m": 0.0, "frame_skip": 1,
            "sample_count": 1, "file": "episode-000001.npz",
        },
        samples=[ReplaySample(0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))],
    )
    with caplog.at_level(logging.WARNING):
        assert writer.submit(payload)
        writer.close()
    assert "Could not persist visual replay episode" in caplog.text


def test_visual_replay_config_is_backward_compatible_and_auto_selects_live_backend() -> None:
    config = TrainingConfig()
    assert config.records_visual_replays is False
    config.backend = "websocket"
    assert config.records_visual_replays is True
    restored = TrainingConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.visual_replay_sample_hz == 20.0
    old_config = config.to_dict()
    old_config.pop("visual_replay_enabled")
    old_config.pop("visual_replay_sample_hz")
    old_config.pop("visual_replay_observations")
    assert TrainingConfig.from_dict(old_config).records_visual_replays
