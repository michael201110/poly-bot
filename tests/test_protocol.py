from __future__ import annotations

import numpy as np
import pytest

from polybot.mock import MockSimulatorTransport
from polybot.protocol import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    Action,
    ProtocolViolation,
    Telemetry,
    request_message,
    response_result,
)


def test_action_policy_round_trip() -> None:
    action = Action.from_policy(np.asarray([0, 1, 0], dtype=np.int64))
    assert action == Action(steer=-1, throttle=True, brake=False)
    np.testing.assert_array_equal(action.to_policy(), [0, 1, 0])
    assert Action.from_wire(action.to_wire()) == action


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([3, 0, 0]),
        np.asarray([1, 2, 0]),
        np.asarray([1, 0]),
        np.asarray([1.5, 0.0, 0.0]),
    ],
)
def test_invalid_policy_actions_are_rejected(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Action.from_policy(value)


def test_mock_telemetry_vector_matches_schema() -> None:
    transport = MockSimulatorTransport()
    hello = request_message(
        0,
        "hello",
        {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "lookahead_count": 6,
        },
    )
    response_result(transport.request(hello), expected_id=0)
    reset = request_message(1, "reset", {"seed": 3, "track_id": "mock/gentle-s"})
    result = response_result(transport.request(reset), expected_id=1)
    telemetry = Telemetry.from_wire(result["state"], lookahead_count=6)
    vector = telemetry.to_vector()

    assert vector.shape == (Telemetry.vector_size(6),)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))


def test_response_id_mismatch_is_rejected() -> None:
    response = {
        "protocol": PROTOCOL_NAME,
        "v": PROTOCOL_VERSION,
        "id": 9,
        "ok": True,
        "result": {},
    }
    with pytest.raises(ProtocolViolation, match="does not match"):
        response_result(response, expected_id=8)


def test_remote_error_is_structured() -> None:
    response = {
        "protocol": PROTOCOL_NAME,
        "v": PROTOCOL_VERSION,
        "id": 2,
        "ok": False,
        "error": {"code": "stale_episode", "message": "reset first"},
    }
    with pytest.raises(ProtocolViolation, match="stale_episode: reset first"):
        response_result(response, expected_id=2)
