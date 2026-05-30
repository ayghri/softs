"""TCP-based transfer medium."""

import logging
import socket
import struct
import threading

from .base import Medium

logger = logging.getLogger(__name__)

# Wire: [4B offset][4B length][data]
_HDR = "!II"
_HDR_SZ = struct.calcsize(_HDR)


class TCPMedium(Medium):
    """buf_name is "host:port"."""

    def __init__(
        self,
        slot_count: int,
        slot_stride: int,
        host: str = "127.0.0.1",
        port: int = 0,
        create: bool = True,
    ):
        self._slot_count = slot_count
        self._slot_stride = slot_stride
        self._owner = create
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._server = None
        self._sock = None
        self._thread = None

        if create:
            self._buffer = bytearray(slot_count * slot_stride)
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((host, port))
            self._server.listen(32)
            self._host, self._port = self._server.getsockname()
            self._thread = threading.Thread(target=self._accept_loop, daemon=True)
            self._thread.start()
        else:
            self._buffer = bytearray()
            self._host, self._port = host, port
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((host, port))

    def _accept_loop(self) -> None:
        assert self._owner
        self._server.settimeout(0.5)  # type: ignore
        while not self._shutdown.is_set():
            try:
                conn, _ = self._server.accept()  # type: ignore
                threading.Thread(
                    target=self._handle, args=(conn,), daemon=True
                ).start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle(self, conn: socket.socket) -> None:
        try:
            while not self._shutdown.is_set():
                hdr = self._recvn(conn, _HDR_SZ)
                if hdr is None:
                    break
                offset, length = struct.unpack(_HDR, hdr)
                data = self._recvn(conn, length)
                if data is None:
                    break
                with self._lock:
                    self._buffer[offset : offset + length] = data
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()

    @staticmethod
    def _recvn(sock: socket.socket, n: int) -> bytes | None:
        parts, left = [], n
        while left > 0:
            chunk = sock.recv(left)
            if not chunk:
                return None
            parts.append(chunk)
            left -= len(chunk)
        return b"".join(parts)

    @property
    def buf_name(self) -> str:
        return f"{self._host}:{self._port}"

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def slot_stride(self) -> int:
        return self._slot_stride

    def write(self, slot_offset: int, data: bytes) -> bool:
        if self._owner:
            # Server-side direct write (used in tests)
            with self._lock:
                self._buffer[slot_offset : slot_offset + len(data)] = data
            return True
        try:
            self._sock.sendall(struct.pack(_HDR, slot_offset, len(data)) + data)
            return True
        except (ConnectionError, OSError):
            return False

    def read(self, slot_id: int) -> bytes:
        off = slot_id * self._slot_stride
        with self._lock:
            return bytes(self._buffer[off : off + self._slot_stride])

    def close(self) -> None:
        self._shutdown.set()
        if self._server:
            self._server.close()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            self._sock.close()

    @classmethod
    def attach(cls, buf_name: str) -> "TCPMedium":
        host, port_str = buf_name.rsplit(":", 1)
        return cls(0, 0, host=host, port=int(port_str), create=False)
