from __future__ import annotations

import json
import socket
from threading import Thread

import pytest
from websockets.sync.client import connect

from polybot.protocol import request_message, success_response
from polybot.transport import WebSocketServerTransport


def unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_websocket_transport_round_trip() -> None:
    transport = WebSocketServerTransport(
        "127.0.0.1",
        unused_loopback_port(),
        connect_timeout_s=2.0,
        request_timeout_s=2.0,
    )
    transport.start()

    def adapter() -> None:
        with connect(transport.endpoint) as websocket:
            request = json.loads(websocket.recv())
            websocket.send(json.dumps(success_response(request, {"pong": True})))

    adapter_thread = Thread(target=adapter, daemon=True)
    adapter_thread.start()
    try:
        request = request_message(4, "ping", {})
        response = transport.request(request)
        assert response["id"] == 4
        assert response["result"] == {"pong": True}
    finally:
        transport.close()
        adapter_thread.join(timeout=2.0)


def test_websocket_transport_rejects_non_loopback_host() -> None:
    try:
        WebSocketServerTransport("0.0.0.0", 8765)
    except ValueError as exc:
        assert "loopback" in str(exc)
    else:
        raise AssertionError("non-loopback listener was accepted")


def test_request_uses_connect_timeout_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = WebSocketServerTransport(
        "127.0.0.1",
        unused_loopback_port(),
        connect_timeout_s=300.0,
        request_timeout_s=0.01,
    )
    connect_arguments: list[float | None] = []

    class FakeConnection:
        def send(self, raw: str) -> None:
            request = json.loads(raw)
            transport._incoming.put(success_response(request, {"ready": True}))

        def close(self, *, code: int, reason: str) -> None:
            del code, reason

    monkeypatch.setattr(
        transport,
        "connect",
        lambda timeout_s=None: connect_arguments.append(timeout_s),
    )
    transport._connection = FakeConnection()  # type: ignore[assignment]
    try:
        response = transport.request(request_message(1, "hello", {}))
    finally:
        transport.close()

    assert response["result"] == {"ready": True}
    assert connect_arguments == [None]
