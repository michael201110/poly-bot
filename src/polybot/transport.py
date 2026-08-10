"""Simulator transport implementations."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Protocol, runtime_checkable

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from polybot.protocol import ProtocolViolation


class TransportClosed(RuntimeError):
    """Raised when a simulator transport is used after it closes."""


@runtime_checkable
class SimulatorTransport(Protocol):
    """Synchronous request/response boundary used by :class:`PolyTrackEnv`."""

    def request(
        self, message: Mapping[str, Any], *, timeout_s: float | None = None
    ) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class WebSocketServerTransport:
    """Accept one local game-adapter connection and exchange JSON messages.

    PolyTrack's renderer can create a native WebSocket client, so the Python
    trainer acts as the server. Only one request may be in flight; this matches
    Gymnasium's synchronous ``step`` contract and avoids ambiguous action order.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        connect_timeout_s: float = 60.0,
        request_timeout_s: float = 10.0,
        max_message_bytes: int = 1_000_000,
    ) -> None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("the training bridge must listen on a loopback address")
        if not 0 < port < 65536:
            raise ValueError("port must be between 1 and 65535")
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self.request_timeout_s = request_timeout_s
        self.max_message_bytes = max_message_bytes

        self._connected = Event()
        self._closed = Event()
        self._request_lock = Lock()
        self._connection_lock = Lock()
        self._connection: ServerConnection | None = None
        self._incoming: Queue[Mapping[str, Any] | BaseException] = Queue()
        self._server: Any = None
        self._thread: Thread | None = None

    @property
    def endpoint(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self) -> None:
        if self._closed.is_set():
            raise TransportClosed("transport is closed")
        if self._server is not None:
            return
        self._server = serve(
            self._handle_connection,
            self.host,
            self.port,
            max_size=self.max_message_bytes,
        )
        self._thread = Thread(
            target=self._server.serve_forever,
            name="polybot-websocket-server",
            daemon=True,
        )
        self._thread.start()

    def connect(self, timeout_s: float | None = None) -> None:
        self.start()
        timeout = self.connect_timeout_s if timeout_s is None else timeout_s
        if not self._connected.wait(timeout):
            raise TimeoutError(
                f"no PolyTrack adapter connected to {self.endpoint} within {timeout}s"
            )

    def _handle_connection(self, connection: ServerConnection) -> None:
        with self._connection_lock:
            if self._connection is not None:
                connection.close(code=1013, reason="a simulator is already connected")
                return
            self._connection = connection
            self._connected.set()

        try:
            for raw_message in connection:
                if not isinstance(raw_message, str):
                    self._incoming.put(
                        ProtocolViolation("binary WebSocket messages are unsupported")
                    )
                    continue
                try:
                    decoded = json.loads(raw_message)
                    if not isinstance(decoded, Mapping):
                        raise ProtocolViolation("WebSocket response must be a JSON object")
                    self._incoming.put(decoded)
                except (json.JSONDecodeError, ProtocolViolation) as exc:
                    self._incoming.put(exc)
        except ConnectionClosed as exc:
            if not self._closed.is_set():
                self._incoming.put(exc)
        finally:
            with self._connection_lock:
                if self._connection is connection:
                    self._connection = None
                    self._connected.clear()

    def request(
        self, message: Mapping[str, Any], *, timeout_s: float | None = None
    ) -> Mapping[str, Any]:
        if self._closed.is_set():
            raise TransportClosed("transport is closed")
        timeout = self.request_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout

        with self._request_lock:
            self.connect(max(0.0, deadline - time.monotonic()))
            with self._connection_lock:
                connection = self._connection
            if connection is None:
                raise ConnectionError("PolyTrack adapter disconnected before the request")
            try:
                connection.send(json.dumps(dict(message), allow_nan=False, separators=(",", ":")))
            except (ConnectionClosed, TypeError, ValueError) as exc:
                raise ConnectionError("could not send request to PolyTrack adapter") from exc

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("simulator request timed out")
            try:
                response = self._incoming.get(timeout=remaining)
            except Empty as exc:
                raise TimeoutError("simulator request timed out") from exc
            if isinstance(response, BaseException):
                raise ConnectionError("PolyTrack adapter connection failed") from response
            return response

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        with self._connection_lock:
            connection = self._connection
            self._connection = None
        if connection is not None:
            try:
                connection.close(code=1000, reason="trainer closed")
            except ConnectionClosed:
                pass
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._connected.clear()
