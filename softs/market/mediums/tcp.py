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
    """Transfer medium backed by a TCP server.

    The owner (client) binds a server socket; its ``address`` is ``"host:port"``.
    Writers (suppliers) ``attach(address)`` and stream ``(offset, data)`` frames
    to the owner, which stores them in its buffer. The owner reads slots from
    that buffer.
    """

    def __init__(
        self,
        address,
        slot_size,
        num_slots,
        create=False,
        host: str = "127.0.0.1",
        **kwargs,
    ):
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._server: socket.socket | None = None
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._conns: set[socket.socket] = set()

        if create:
            self._buffer = bytearray(slot_size * num_slots)
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            bind_host, bind_port = host, 0
            if address:
                bind_host, port_str = address.rsplit(":", 1)
                bind_port = int(port_str)
            self._server.bind((bind_host, bind_port))
            self._server.listen(32)
            bound_host, bound_port = self._server.getsockname()
            address = f"{bound_host}:{bound_port}"
            self._thread = threading.Thread(target=self._accept_loop, daemon=True)
            self._thread.start()
        else:
            if address is None:
                raise ValueError("address required when create=False")
            self._buffer = bytearray()
            conn_host, port_str = address.rsplit(":", 1)
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((conn_host, int(port_str)))

        super().__init__(address, slot_size, num_slots, create)

    def _accept_loop(self) -> None:
        assert self._server is not None
        self._server.settimeout(0.5)
        while not self._shutdown.is_set():
            try:
                conn, _ = self._server.accept()
                threading.Thread(
                    target=self._handle, args=(conn,), daemon=True
                ).start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(0.5)
        with self._lock:
            self._conns.add(conn)
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
                # Ack so the writer's write() returns only after the data is
                # stored, giving the reader a happens-before guarantee.
                conn.sendall(b"\x01")
        except (ConnectionError, OSError):
            pass
        finally:
            with self._lock:
                self._conns.discard(conn)
            conn.close()

    def _recvn(self, sock: socket.socket, n: int) -> bytes | None:
        """Read exactly n bytes. Returns None on EOF or shutdown.

        Tolerates the per-socket timeout so handler threads observe shutdown
        instead of blocking forever when a peer crashes mid-stream.
        """
        parts, left = [], n
        while left > 0:
            if self._shutdown.is_set():
                return None
            try:
                chunk = sock.recv(left)
            except socket.timeout:
                continue
            if not chunk:
                return None
            parts.append(chunk)
            left -= len(chunk)
        return b"".join(parts)

    def write(self, slot_offset: int, data: bytes) -> bool:
        if self.is_owner:
            # Server-side direct write (used in tests).
            with self._lock:
                self._buffer[slot_offset : slot_offset + len(data)] = data
            return True
        try:
            assert self._sock is not None
            self._sock.sendall(struct.pack(_HDR, slot_offset, len(data)) + data)
            # Wait for the owner's ack so the data is stored before we return.
            ack = self._sock.recv(1)
            return ack == b"\x01"
        except (ConnectionError, OSError):
            logger.warning(
                "tcp write failed at offset %d (len %d) on '%s'",
                slot_offset,
                len(data),
                self.address,
                exc_info=True,
            )
            return False

    def read(self, slot_id: int) -> bytes:
        off = slot_id * self.slot_size
        with self._lock:
            return bytes(self._buffer[off : off + self.slot_size])

    @classmethod
    def attach(cls, address: str) -> "TCPMedium":
        return cls(address=address, slot_size=0, num_slots=0, create=False)

    def close(self) -> None:
        self._shutdown.set()
        if self._server:
            self._server.close()
        if self._thread:
            self._thread.join(timeout=2.0)
        with self._lock:
            conns = list(self._conns)
        for c in conns:
            try:
                c.close()
            except OSError:
                pass
        if self._sock:
            self._sock.close()
