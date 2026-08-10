from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from polybot import cli
from polybot.protocol import ProtocolViolation


@pytest.mark.parametrize(
    ("backend", "track", "frame_skip", "max_steps"),
    [
        ("mock", "mock/gentle-s", 4, 2_000),
        ("websocket", "current", 10, 30_000),
    ],
)
def test_backend_defaults(backend: str, track: str, frame_skip: int, max_steps: int) -> None:
    args = argparse.Namespace(backend=backend, track=None, frame_skip=None, max_steps=None)

    cli._apply_backend_defaults(args)

    assert args.track == track
    assert args.frame_skip == frame_skip
    assert args.max_steps == max_steps


def test_drive_uses_real_game_defaults(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    created: dict[str, object] = {}

    class FakeTransport:
        def close(self) -> None:
            created["transport_closed"] = True

    class FakeEnvironment:
        def __init__(self, transport: object, **kwargs: object) -> None:
            created["transport"] = transport
            created["kwargs"] = kwargs
            self.latest_telemetry = object()

        def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
            created["seed"] = seed
            return np.zeros(1, dtype=np.float32), {}

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
            created["action"] = action
            return (
                np.zeros(1, dtype=np.float32),
                1.25,
                True,
                False,
                {
                    "events": ("finish",),
                    "elapsed_s": 12.5,
                    "route_progress_m": 100.0,
                },
            )

        def close(self) -> None:
            created["environment_closed"] = True

    class FakeController:
        def __init__(self, *, steering_sign: int, max_speed_mps: float) -> None:
            created["steering_sign"] = steering_sign
            created["max_speed_mps"] = max_speed_mps

        def policy_action(self, telemetry: object) -> np.ndarray:
            assert telemetry is not None
            return np.asarray([1, 1, 0], dtype=np.int64)

    transport = FakeTransport()
    monkeypatch.setattr(cli, "_make_transport", lambda args: transport)
    monkeypatch.setattr(cli, "PolyTrackEnv", FakeEnvironment)
    monkeypatch.setattr(cli, "CenterlineController", FakeController)

    assert cli.drive_main([]) == 0

    assert created["kwargs"] == {
        "track_id": "current",
        "lookahead_count": 12,
        "frame_skip": 10,
        "max_episode_steps": 30_000,
        "request_timeout_s": 60.0,
    }
    assert created["seed"] == 0
    assert created["steering_sign"] == 1
    assert created["max_speed_mps"] == 18.0
    np.testing.assert_array_equal(created["action"], [1, 1, 0])
    assert created["environment_closed"] is True
    assert '"finished": true' in capsys.readouterr().out


def test_drive_rejects_stochastic_centerline_policy() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.drive_main(["--stochastic"])


def test_drive_waits_for_race_and_reference(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    attempts = iter(
        [
            ProtocolViolation("game_not_ready: enter the race"),
            ProtocolViolation("missing_reference: load a ghost"),
            (np.zeros(1, dtype=np.float32), {}),
        ]
    )

    class FakeEnvironment:
        def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
            assert seed == 7
            result = next(attempts)
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    observation, info = cli._reset_drive_when_ready(
        FakeEnvironment(),  # type: ignore[arg-type]
        seed=7,
        timeout_s=30,
    )

    np.testing.assert_array_equal(observation, np.zeros(1, dtype=np.float32))
    assert info == {}
    output = capsys.readouterr().out
    assert "enter the race" in output
    assert "load a ghost" in output


def test_drive_does_not_expose_an_endpoint_the_mod_cannot_use() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.drive_main(["--port", "9999"])


def test_websocket_transport_uses_fixed_public_mod_endpoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[str] = []

    class FakeWebSocketTransport:
        endpoint = "ws://127.0.0.1:8765"

        def __init__(self, *args: object, **kwargs: object) -> None:
            events.append("transport created")

        def request(self, message: object, *, timeout_s: float | None = None) -> object:
            return message

        def close(self) -> None:
            events.append("transport closed")

    monkeypatch.setattr(cli, "WebSocketServerTransport", FakeWebSocketTransport)
    args = SimpleNamespace(
        backend="websocket",
        host="127.0.0.1",
        port=8765,
        connect_timeout=3.0,
        request_timeout=4.0,
    )

    transport = cli._make_transport(args)
    assert events == ["transport created"]
    output = capsys.readouterr().out
    assert "Waiting for the local PolyTrack mod at ws://127.0.0.1:8765" in output

    transport.close()
    assert events == ["transport created", "transport closed"]


def test_drive_closes_owned_transport_if_environment_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    class FakeTransport:
        def close(self) -> None:
            nonlocal closed
            closed = True

    def fail_environment(*args: object, **kwargs: object) -> None:
        raise RuntimeError("environment failed")

    monkeypatch.setattr(cli, "_make_transport", lambda args: FakeTransport())
    monkeypatch.setattr(cli, "PolyTrackEnv", fail_environment)

    with pytest.raises(RuntimeError, match="environment failed"):
        cli.drive_main([])
    assert closed is True
