from __future__ import annotations

from typing import Any

from polybot.mock import MockSimulatorTransport
from polybot.protocol import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    ProtocolViolation,
    request_message,
    response_result,
)


def handshake_and_reset(transport: MockSimulatorTransport) -> dict[str, Any]:
    hello = request_message(
        0,
        "hello",
        {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "lookahead_count": 4,
        },
    )
    response_result(transport.request(hello), expected_id=0)
    reset = request_message(1, "reset", {"seed": 0, "track_id": "mock/straight"})
    return dict(response_result(transport.request(reset), expected_id=1))


def test_stale_episode_is_rejected() -> None:
    transport = MockSimulatorTransport()
    reset_result = handshake_and_reset(transport)
    step = request_message(
        2,
        "step",
        {
            "episode_id": reset_result["episode_id"] + "-stale",
            "action": {"steer": 0, "throttle": 1, "brake": 0},
            "ticks": 1,
        },
    )
    try:
        response_result(transport.request(step), expected_id=2)
    except ProtocolViolation as exc:
        assert "episode_id is stale" in str(exc)
    else:
        raise AssertionError("stale episode was accepted")


def test_step_before_reset_is_rejected_without_crashing_peer() -> None:
    transport = MockSimulatorTransport()
    hello = request_message(
        0,
        "hello",
        {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "lookahead_count": 4,
        },
    )
    response_result(transport.request(hello), expected_id=0)
    step = request_message(
        1,
        "step",
        {
            "episode_id": "missing",
            "action": {"steer": 0, "throttle": 1, "brake": 0},
            "ticks": 1,
        },
    )
    try:
        response_result(transport.request(step), expected_id=1)
    except ProtocolViolation as exc:
        assert "reset must be called" in str(exc)
    else:
        raise AssertionError("step before reset was accepted")

    # A structured error must not kill the simulator.
    reset = request_message(2, "reset", {"seed": 0, "track_id": "mock/straight"})
    result = response_result(transport.request(reset), expected_id=2)
    assert result["tick"] == 0


def test_close_is_idempotent() -> None:
    transport = MockSimulatorTransport()
    transport.close()
    transport.close()
