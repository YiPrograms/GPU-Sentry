"""Length-prefixed JSON framing shared by online processor tests and runtime."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any


HEADER_SIZE = 4


class FrameError(RuntimeError):
    """Raised when a framed stream is malformed."""


def recv_frame(conn: socket.socket, max_bytes: int) -> dict[str, Any] | None:
    header = _recv_exact(conn, HEADER_SIZE)
    if header is None:
        return None
    size = struct.unpack(">I", header)[0]
    if size > max_bytes:
        raise FrameError(f"frame too large: {size} > {max_bytes}")
    payload = _recv_exact(conn, size)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def send_frame(conn: socket.socket, message: dict[str, Any]) -> None:
    payload = json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_exact(conn: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
