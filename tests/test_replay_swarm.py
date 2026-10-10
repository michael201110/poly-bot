from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from polybot.replay_swarm import (
    ColorScale,
    ColorStop,
    ReplayPlaybackOptions,
    ReplaySelection,
    SelectedReplayPayload,
    filter_replays,
    interpolate_position,
    interpolate_quaternion,
    interpolate_replay,
    main,
    parse_color_stops,
    replay_sample_chunks,
    require_single_replay,
    resolve_replay_directories,
    select_replays,
    send_replay_playback,
    send_replay_swarm,
    stratified_sample,
    swarm_colors,
)
from polybot.training.visual_replays import INDEX_SCHEMA, ReplayPayload, ReplaySample


def entry(
    episode: int,
    step: int,
    status: str = "failed",
    *,
    run_id: str = "run",
) -> dict:
    return {
        "run_id": run_id,
        "episode_id": f"episode-{episode:06d}",
        "algorithm": "tqc",
        "track_id": "track",
        "track_name": "Track",
        "training_step": step,
        "training_step_start": step,
        "training_step_end": step + 10,
        "episode_length_decisions": 10,
        "episode_length_ticks": 300,
        "status": status,
        "final_progress_m": float(step),
        "frame_skip": 30,
        "sample_count": 2,
        "file": f"episode-{episode:06d}.npz",
    }


def test_index_step_and_episode_ranges_are_inclusive_and_use_start_step() -> None:
    entries = [
        entry(1, 0),
        entry(2, 10, "finished"),
        entry(3, 20, "timeout"),
        entry(4, 25_000, "finished"),
        entry(5, 25_001),
    ]
    selected = filter_replays(entries, steps=(10, 25_000), episodes=(2, 4))
    assert [item["episode_id"] for item in selected] == [
        "episode-000002",
        "episode-000003",
        "episode-000004",
    ]
    assert filter_replays(entries, steps=(10, 10), finished_only=True) == [entries[1]]
    assert filter_replays(entries, failed_only=True) == [entries[0], entries[2], entries[4]]
    with pytest.raises(ValueError, match="cannot both"):
        filter_replays(entries, finished_only=True, failed_only=True)


def test_index_load_does_not_require_or_open_npz_payloads(tmp_path) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    # Index records deliberately point to absent files; selection must not inspect them.
    metadata = [entry(index + 1, (index + 1) * 1000) for index in range(4)]
    (directory / "index.json").write_text(
        json.dumps({"schema": INDEX_SCHEMA, "run": {"run_id": "run"}, "episodes": metadata}),
        encoding="utf-8",
    )
    total, selected = select_replays([directory], steps=(1000, 3000), max_cars=5)
    assert total == 4
    assert [item.training_step for item in selected] == [1000, 2000, 3000]


def test_seeded_stratified_sampling_is_deterministic_and_spans_age_range() -> None:
    entries = [entry(number, (number - 1) * 1000) for number in range(1, 101)]
    first = stratified_sample(entries, 10, seed=71)
    again = stratified_sample(entries, 10, seed=71)
    other_seed = stratified_sample(entries, 10, seed=72)
    assert [item["episode_id"] for item in first] == [item["episode_id"] for item in again]
    assert [item["episode_id"] for item in first] != [item["episode_id"] for item in other_seed]
    ages = [item["training_step_start"] for item in first]
    assert len(set(ages)) == 10
    assert min(ages) < 15_000
    assert max(ages) > 80_000


def test_color_scale_interpolates_clamps_and_supports_custom_stops() -> None:
    scale = ColorScale(0, 1_000_000)
    assert scale.color(0) == (255, 0, 0)
    assert scale.color(125_000) == (255, 64, 0)
    assert scale.color(250_000) == (255, 128, 0)
    assert scale.color(500_000) == (255, 255, 0)
    assert scale.color(750_000) == (128, 255, 0)
    assert scale.color(1_000_000) == (0, 255, 0)
    assert scale.color(-1) == scale.color(0)
    assert scale.color(2_000_000) == scale.color(1_000_000)
    assert scale.hex_color(500_000) == "#ffff00"

    stops = parse_color_stops(["100:#000000", "200:#ffffff"])
    custom = ColorScale(stops=stops)
    assert custom.color(150) == (128, 128, 128)
    assert custom.color(0) == (0, 0, 0)
    assert custom.color(300) == (255, 255, 255)
    assert custom.bucket(100) == "#000000"
    with pytest.raises(ValueError, match="distinct"):
        ColorScale(stops=(ColorStop(0, "red"), ColorStop(0, "green")))


def test_position_and_elapsed_transform_interpolation() -> None:
    samples = [
        ReplaySample(0, 5.0, (0.0, 1.0, 2.0), (0.0, 0.0, 0.0, 1.0)),
        ReplaySample(100, 7.0, (10.0, 3.0, 6.0), (0.0, 0.0, 1.0, 0.0)),
    ]
    assert interpolate_position(samples[0].position_m, samples[1].position_m, 0.5) == (
        5.0,
        2.0,
        4.0,
    )
    transform = interpolate_replay(samples, 1.0)
    assert transform is not None
    assert transform.elapsed_s == 6.0
    assert transform.position_m == (5.0, 2.0, 4.0)
    assert transform.opacity == 1.0


def test_quaternion_interpolation_uses_shortest_arc_slerp() -> None:
    halfway = interpolate_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 0.0),
        0.5,
    )
    assert halfway == pytest.approx((0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    antipodal = interpolate_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, -1.0),
        0.5,
    )
    assert antipodal == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert np.linalg.norm(halfway) == pytest.approx(1.0)


def test_short_and_long_episodes_have_independent_end_behaviours() -> None:
    short = [
        ReplaySample(0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ReplaySample(10, 1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ]
    long = short + [
        ReplaySample(20, 2.0, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ]

    assert interpolate_replay(short, 1.5, end_behavior="disappear") is None
    frozen = interpolate_replay(short, 2.0, end_behavior="freeze")
    assert frozen is not None and frozen.position_m == (1.0, 0.0, 0.0) and frozen.opacity == 1.0
    fading = interpolate_replay(short, 1.5, end_behavior="fade", fade_duration_s=1.0)
    assert fading is not None and fading.position_m == (1.0, 0.0, 0.0)
    assert fading.opacity == pytest.approx(0.5)
    assert interpolate_replay(short, 2.0, end_behavior="fade", fade_duration_s=1.0) is None
    later = interpolate_replay(long, 1.5)
    assert later is not None and later.position_m == (1.5, 0.0, 0.0)


def test_live_playback_selection_is_capped_at_one_replay() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        require_single_replay([], max_cars=1)
    with pytest.raises(ValueError, match="exactly one selected replay"):
        require_single_replay(
            [ReplaySelection(Path("run"), entry(1, 0))],
            max_cars=2,
        )


def test_playback_options_validate_and_normalize_visual_settings() -> None:
    assert ReplayPlaybackOptions().opacity == 1.0
    assert ReplayPlaybackOptions(color="red").color == "#ff0000"
    for options in (
        {"speed": 0.09},
        {"speed": 8.1},
        {"opacity": -0.1},
        {"opacity": 1.1},
        {"color": "not-a-color"},
        {"end_behavior": "teleport"},
        {"fade_duration_s": 11},
    ):
        with pytest.raises(ValueError):
            ReplayPlaybackOptions(**options)


def test_replay_transfer_chunks_preserve_timestamps_and_validate_order() -> None:
    samples = [
        ReplaySample(
            tick=index * 30,
            elapsed_s=index * 0.03,
            position_m=(float(index), 2.0, 3.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        for index in range(300)
    ]
    chunks = list(replay_sample_chunks(samples))
    assert [chunk["start"] for chunk in chunks] == [0, 256]
    assert len(chunks[0]["samples"]) == 256
    assert chunks[1]["samples"][0][0] == 7680
    assert chunks[1]["samples"][0][1] == pytest.approx(7.68)
    assert chunks[1]["samples"][0][2:] == [256.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="ordered"):
        list(replay_sample_chunks([samples[1], samples[0]]))
    with pytest.raises(ValueError, match="from 1 to"):
        list(replay_sample_chunks([]))
    with pytest.raises(ValueError, match="quaternion cannot be zero"):
        ReplaySample(0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))


def test_replay_transfer_rejects_oversized_payload_without_iterating_it() -> None:
    sample = ReplaySample(0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="500000"):
        list(replay_sample_chunks([sample] * 500_001))


def test_live_playback_uses_existing_v2_bridge_and_chunks_without_changing_training_ops() -> None:
    class RecordingTransport:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        def request(self, message):
            self.messages.append(dict(message))
            return {
                "protocol": "polybot.sim",
                "v": 2,
                "id": message["id"],
                "ok": True,
                "result": {"accepted": True, "protocol_version": 2},
            }

    samples = [
        ReplaySample(index, index * 0.001, (float(index), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)) for index in range(300)
    ]
    transport = RecordingTransport()
    send_replay_playback(
        transport,
        action="play",
        payload=ReplayPayload(metadata={}, samples=samples),
        options=ReplayPlaybackOptions(color="#123456"),
    )
    assert [message["op"] for message in transport.messages] == [
        "hello",
        "visual_replay_begin",
        "visual_replay_chunk",
        "visual_replay_chunk",
        "visual_replay_commit",
    ]
    assert transport.messages[0]["params"] == {
        "protocol": "polybot.sim",
        "protocol_version": 2,
        "lookahead_count": 1,
    }
    assert transport.messages[1]["params"]["sample_count"] == 300
    assert transport.messages[-1]["params"] == {"autoplay": True}

    with pytest.raises(ValueError, match="seek requires"):
        send_replay_playback(
            transport,
            action="seek",
            payload=None,
            options=ReplayPlaybackOptions(),
        )


def test_live_playback_controls_use_v2_requests_and_preserve_existing_operations() -> None:
    class RecordingTransport:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        def request(self, message):
            self.messages.append(dict(message))
            return {
                "protocol": "polybot.sim",
                "v": 2,
                "id": message["id"],
                "ok": True,
                "result": {"protocol_version": 2},
            }

    transport = RecordingTransport()
    send_replay_playback(
        transport,
        action="seek",
        payload=None,
        options=ReplayPlaybackOptions(),
        seek_seconds=1.25,
        update_settings=("speed", "opacity", "color"),
    )
    assert [item["op"] for item in transport.messages] == [
        "hello",
        "visual_replay_seek",
        "visual_replay_speed",
        "visual_replay_opacity",
        "visual_replay_color",
    ]
    assert transport.messages[1]["params"] == {"seconds": 1.25}
    assert transport.messages[2]["params"] == {"value": 1.0}


def test_configure_action_updates_playback_without_changing_play_state() -> None:
    class RecordingTransport:
        def __init__(self) -> None:
            self.operations: list[dict] = []

        def request(self, message):
            self.operations.append(dict(message))
            return {
                "protocol": "polybot.sim",
                "v": 2,
                "id": message["id"],
                "ok": True,
                "result": {"protocol_version": 2},
            }

    transport = RecordingTransport()
    send_replay_playback(
        transport,
        action="configure",
        payload=None,
        options=ReplayPlaybackOptions(speed=2.0, opacity=0.5, color="#00ff00"),
        update_settings=("speed", "opacity", "color"),
    )
    assert [item["op"] for item in transport.operations] == [
        "hello",
        "visual_replay_speed",
        "visual_replay_opacity",
        "visual_replay_color",
    ]
    with pytest.raises(ValueError, match="at least one"):
        send_replay_playback(
            transport,
            action="configure",
            payload=None,
            options=ReplayPlaybackOptions(),
        )


@pytest.mark.parametrize(
    ("action", "operation"),
    [
        ("resume", "visual_replay_play"),
        ("pause", "visual_replay_pause"),
        ("restart", "visual_replay_restart"),
        ("clear", "visual_replay_clear"),
    ],
)
def test_playback_lifecycle_commands_are_additive_v2_operations(action: str, operation: str) -> None:
    class RecordingTransport:
        def __init__(self) -> None:
            self.operations: list[str] = []

        def request(self, message):
            self.operations.append(message["op"])
            return {
                "protocol": "polybot.sim",
                "v": 2,
                "id": message["id"],
                "ok": True,
                "result": {"protocol_version": 2},
            }

    transport = RecordingTransport()
    send_replay_playback(
        transport,
        action=action,
        payload=None,
        options=ReplayPlaybackOptions(),
    )
    assert transport.operations == ["hello", operation]


def test_single_sample_episode_obeys_endpoint_behaviour() -> None:
    samples = [ReplaySample(0, 0.0, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))]
    assert interpolate_replay(samples, 0.0) is not None
    assert interpolate_replay(samples, 0.1, end_behavior="disappear") is None
    frozen = interpolate_replay(samples, 5.0, end_behavior="freeze")
    assert frozen is not None and frozen.position_m == (2.0, 0.0, 0.0)
    fading = interpolate_replay(samples, 0.25, end_behavior="fade", fade_duration_s=0.5)
    assert fading is not None and fading.opacity == pytest.approx(0.5)


def test_malformed_index_metadata_and_missing_replays(tmp_path, capsys) -> None:
    assert resolve_replay_directories(tmp_path / "no-replays") == []
    assert main(["--run", str(tmp_path / "no-replays"), "--steps", "0:25000", "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["total_indexed_episodes"] == 0
    assert report["selected_episode_count"] == 0
    assert report["selected_step_range"] is None

    corrupt = tmp_path / "corrupt-run"
    corrupt.mkdir()
    (corrupt / "index.json").write_text("{broken", encoding="utf-8")
    assert main(["--run", str(corrupt), "--dry-run"]) == 2
    assert "cannot read replay index" in capsys.readouterr().err

    directory = tmp_path / "bad-run"
    directory.mkdir()
    (directory / "index.json").write_text(
        json.dumps({"schema": INDEX_SCHEMA, "episodes": [{"training_step_start": "bad"}]}),
        encoding="utf-8",
    )
    assert main(["--run", str(directory), "--dry-run"]) == 2
    assert "metadata" in capsys.readouterr().err


def test_cli_selection_report_for_hypothetical_step_window(tmp_path, capsys) -> None:
    directory = tmp_path / "visual_replays" / "run-123"
    directory.mkdir(parents=True)
    metadata = [
        entry(1, 0, "failed"),
        entry(2, 12_500, "finished"),
        entry(3, 25_000, "timeout"),
        entry(4, 25_001, "finished"),
    ]
    (directory / "index.json").write_text(
        json.dumps({"schema": INDEX_SCHEMA, "episodes": metadata}),
        encoding="utf-8",
    )
    code = main(
        [
            "--run",
            str(directory.parent),
            "--steps",
            "0:25000",
            "--max-cars",
            "10",
            "--color-min-step",
            "0",
            "--color-max-step",
            "1000000",
            "--dry-run",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["total_indexed_episodes"] == 4
    assert report["episodes_matching_filters"] == 3
    assert report["selected_episode_count"] == 3
    assert report["selected_step_range"] == [0, 25_000]
    assert report["training_step_median"] == 12_500
    assert report["finish_count"] == 1
    assert report["failure_count"] == 2
    assert [item["episode_id"] for item in report["selected_episodes"]] == [
        "episode-000001",
        "episode-000002",
        "episode-000003",
    ]


class _RecordingTransport:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def request(self, message):
        self.messages.append(dict(message))
        return {
            "protocol": "polybot.sim",
            "v": 2,
            "id": message["id"],
            "ok": True,
            "result": {"protocol_version": 2},
        }


def _selected_payload(episode_number: int, step: int) -> SelectedReplayPayload:
    metadata = entry(episode_number, step)
    selection = ReplaySelection(Path("replays") / "run", metadata)
    samples = [
        ReplaySample(0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ReplaySample(1, 0.03, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ]
    return SelectedReplayPayload(selection, ReplayPayload(metadata={}, samples=samples), "#ff0000")


def test_swarm_transfer_sends_all_episodes_with_individual_age_colors_once() -> None:
    transport = _RecordingTransport()
    payloads = [_selected_payload(1, 0), _selected_payload(2, 250_000)]
    payloads[1] = SelectedReplayPayload(payloads[1].selection, payloads[1].payload, "#ff8000")
    result = send_replay_swarm(
        transport,
        action="play",
        payloads=payloads,
        options=ReplayPlaybackOptions(),
    )
    operations = [message["op"] for message in transport.messages]
    assert operations == [
        "hello",
        "visual_replay_swarm_begin",
        "visual_replay_swarm_episode_begin",
        "visual_replay_swarm_chunk",
        "visual_replay_swarm_episode_commit",
        "visual_replay_swarm_episode_begin",
        "visual_replay_swarm_chunk",
        "visual_replay_swarm_episode_commit",
        "visual_replay_swarm_commit",
    ]
    begin = transport.messages[1]["params"]
    assert begin["ghost_count"] == 2
    assert begin["total_sample_count"] == 4
    episodes = [
        message["params"] for message in transport.messages if message["op"] == "visual_replay_swarm_episode_begin"
    ]
    assert [(item["training_step_start"], item["color"]) for item in episodes] == [
        (0, "#ff0000"),
        (250_000, "#ff8000"),
    ]
    assert transport.messages[-1]["params"] == {"autoplay": True}
    assert result["protocol_version"] == 2


@pytest.mark.parametrize(
    ("action", "operation"),
    [
        ("resume", "visual_replay_play"),
        ("pause", "visual_replay_pause"),
        ("restart", "visual_replay_restart"),
        ("clear", "visual_replay_clear"),
        ("status", "visual_replay_status"),
    ],
)
def test_swarm_controls_use_existing_bridge(action: str, operation: str) -> None:
    transport = _RecordingTransport()
    send_replay_swarm(transport, action=action, options=ReplayPlaybackOptions())
    assert [message["op"] for message in transport.messages] == ["hello", operation]


def test_swarm_preflight_rejects_duplicate_episodes_and_resource_limits(monkeypatch) -> None:
    item = _selected_payload(1, 0)
    transport = _RecordingTransport()
    with pytest.raises(ValueError, match="duplicate"):
        send_replay_swarm(
            transport,
            action="play",
            payloads=[item, item],
            options=ReplayPlaybackOptions(),
        )
    assert not transport.messages

    monkeypatch.setattr("polybot.replay_swarm.MAX_REPLAY_GHOSTS", 1)
    with pytest.raises(ValueError, match="ghost count"):
        send_replay_swarm(
            transport,
            action="play",
            payloads=[item, _selected_payload(2, 10)],
            options=ReplayPlaybackOptions(),
        )
    monkeypatch.setattr("polybot.replay_swarm.MAX_REPLAY_GHOSTS", 500)
    monkeypatch.setattr("polybot.replay_swarm.MAX_REPLAY_TOTAL_SAMPLES", 3)
    with pytest.raises(ValueError, match="total sample count"):
        send_replay_swarm(
            transport,
            action="play",
            payloads=[item, _selected_payload(2, 10)],
            options=ReplayPlaybackOptions(),
        )
    monkeypatch.setattr("polybot.replay_swarm.MAX_REPLAY_TOTAL_SAMPLES", 1_000_000)
    monkeypatch.setattr("polybot.replay_swarm.MAX_REPLAY_PAYLOAD_BYTES", 1)
    with pytest.raises(ValueError, match="payload size"):
        send_replay_swarm(
            transport,
            action="play",
            payloads=[item],
            options=ReplayPlaybackOptions(),
        )
    assert not transport.messages


def test_swarm_sender_rejects_missing_episodes_for_load_and_bad_seek() -> None:
    transport = _RecordingTransport()
    with pytest.raises(ValueError, match="at least one"):
        send_replay_swarm(transport, action="load", options=ReplayPlaybackOptions())
    with pytest.raises(ValueError, match="seek requires"):
        send_replay_swarm(transport, action="seek", options=ReplayPlaybackOptions())


def test_color_scale_supports_single_checkpoint_ranges():
    scale = ColorScale(4378332, 4378333)
    assert scale.hex_color(4378332) == "#ff0000"
    assert scale.hex_color(4378333) == "#00ff00"


def test_same_checkpoint_swarm_has_same_training_step_color(tmp_path):
    selected = [ReplaySelection(tmp_path, entry(index, 123, "finished")) for index in range(1, 6)]
    colors = swarm_colors(selected, ColorScale())
    assert colors == [ColorScale().hex_color(123)] * 5
    assert colors == swarm_colors(selected, ColorScale())


def test_swarm_transfers_hud_only_for_fixed_best_run() -> None:
    transport = _RecordingTransport()
    payloads = [_selected_payload(1, 0), _selected_payload(2, 0)]
    for item, lap in zip(payloads, (23.0, 22.6), strict=True):
        item.selection.metadata["lap_time_s"] = lap
        item.payload.hud_frames = [{
            "schema": "polybot.ai-overlay-frame.v1",
            "elapsed_simulation_s": 0.0,
            "episode": item.episode_key,
        }]
    send_replay_swarm(transport, action="load", payloads=payloads, options=ReplayPlaybackOptions())
    chunks = [message["params"] for message in transport.messages if message["op"] == "visual_replay_swarm_chunk"]
    assert "hud_frames" not in chunks[0]
    assert chunks[1]["hud_frames"][0]["episode"] == payloads[1].episode_key
    assert all(item.payload.hud_frames for item in payloads)
