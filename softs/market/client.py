"""Client for requesting samples from the broker."""

import atexit
import logging
import threading
from collections import deque

import zmq

from .protocol import ClientCmd, decode_payload, make_request
from .mediums import Medium

logger = logging.getLogger(__name__)


class Client:
    _ctx: zmq.Context
    _market: zmq.Socket

    def __init__(
        self,
        endpoint: str,
        medium_cls: type[Medium],
        slot_size: int,
        num_slots: int,
        address: str | None = None,
        send_timeout_ms: int = 5000,
    ):
        self.endpoint = endpoint
        self.slot_count = num_slots
        self.send_timeout_ms = send_timeout_ms
        self._peer_id: int | None = None

        self.medium = medium_cls(
            address=address,
            slot_size=slot_size,
            num_slots=num_slots,
            create=True,
        )
        self._closed = False
        atexit.register(self._atexit_cleanup)

        self._free_slots: deque[int] = deque(range(num_slots))
        self._pending_slots: dict[str, int] = {}
        self._completed_slots: deque[int] = deque()
        self._slot_lock = threading.Lock()

        self._order_counter = 0
        self._order_lock = threading.Lock()
        self._shutdown = threading.Event()

        self._connect()

    def _connect(self):
        self._ctx = zmq.Context()
        self._ctx.linger = 0
        self._market = self._ctx.socket(zmq.DEALER)
        self._market.setsockopt(zmq.LINGER, 0)
        self._market.connect(self.endpoint)

    def _next_order_id(self) -> str:
        with self._order_lock:
            self._order_counter += 1
            return f"c{self._peer_id}-{self._order_counter}"

    def _drain_stale_replies(self) -> None:
        """Consume any messages sitting in the socket buffer (stale replies from timeouts)."""
        while self._market.poll(0):
            frames = self._market.recv_multipart()
            if len(frames) >= 2 and frames[0] == ClientCmd.FULFILLED:
                token = (
                    decode_payload(frames[1]).get("order_id")
                    if frames[1]
                    else None
                )
                if token:
                    with self._slot_lock:
                        slot_id = self._pending_slots.pop(token, None)
                        if slot_id is not None:
                            self._completed_slots.append(slot_id)

    def _send(
        self, cmd: bytes, payload: dict, timeout_ms: int | None = None
    ) -> dict:
        """Send request and wait for reply. Returns error dict on timeout."""
        self._drain_stale_replies()
        timeout = timeout_ms or self.send_timeout_ms
        self._market.send_multipart(make_request(cmd, payload))
        if self._market.poll(timeout):
            return decode_payload(self._market.recv_multipart()[-1])
        return {"ok": False, "error": "timeout"}

    def hello(self) -> dict:
        reply = self._send(ClientCmd.HELLO, {})
        if not reply.get("ok"):
            raise RuntimeError(f"HELLO failed: {reply.get('error')}")
        self._peer_id = reply["peer_id"]
        return reply

    def request_slot(self, product_id: str) -> str | None:
        with self._slot_lock:
            if not self._free_slots:
                return None
            slot_id = self._free_slots.popleft()
            slot_offset = slot_id * self.medium.slot_size

        order_id = self._next_order_id()
        reply = self._send(
            ClientCmd.ORDER,
            {
                "order_id": order_id,
                "product_id": product_id,
                "address": self.medium.address,
                "offset": slot_offset,
            },
        )
        if not reply.get("ok"):
            with self._slot_lock:
                self._free_slots.append(slot_id)
            return None

        with self._slot_lock:
            self._pending_slots[order_id] = slot_id
        return order_id

    def poll_completions(self, timeout_ms: int = 0) -> int:
        count = 0
        while True:
            if not self._market.poll(timeout_ms if count == 0 else 0):
                break
            frames = self._market.recv_multipart()
            if len(frames) >= 2 and frames[0] == ClientCmd.FULFILLED:
                token = (
                    decode_payload(frames[1]).get("order_id")
                    if frames[1]
                    else None
                )
                if token:
                    with self._slot_lock:
                        slot_id = self._pending_slots.pop(token, None)
                        if slot_id is not None:
                            self._completed_slots.append(slot_id)
                    count += 1
        return count

    def _drain_completions(self) -> None:
        poller = zmq.Poller()
        poller.register(self._market, zmq.POLLIN)
        while dict(poller.poll(0)):
            frames = self._market.recv_multipart()
            if len(frames) >= 2 and frames[0] == ClientCmd.FULFILLED:
                token = (
                    decode_payload(frames[1]).get("order_id")
                    if frames[1]
                    else None
                )
                if token:
                    with self._slot_lock:
                        slot_id = self._pending_slots.pop(token, None)
                        if slot_id is not None:
                            self._free_slots.append(slot_id)

    def get_completed(self) -> int | None:
        with self._slot_lock:
            return (
                self._completed_slots.popleft()
                if self._completed_slots
                else None
            )

    def release_slot(self, slot_id: int) -> None:
        with self._slot_lock:
            self._free_slots.append(slot_id)

    def request_sample(
        self, product_id: str, timeout_ms: int = 1000
    ) -> int | None:
        slot_id = self.get_completed()
        if slot_id is not None:
            return slot_id
        self.request_slot(product_id)
        self.poll_completions(timeout_ms)
        return self.get_completed()

    def discard(self) -> int:
        reply = self._send(ClientCmd.DISCARD, {})
        cancelled = reply.get("cancelled", 0)
        with self._slot_lock:
            for slot_id in self._pending_slots.values():
                self._free_slots.append(slot_id)
            self._pending_slots.clear()
            self._completed_slots.clear()
        self._drain_completions()
        return cancelled

    def cancel(self, order_id: str) -> bool:
        reply = self._send(ClientCmd.CANCEL, {"order_id": order_id})
        if reply.get("ok"):
            with self._slot_lock:
                slot_id = self._pending_slots.pop(order_id, None)
                if slot_id is not None:
                    self._free_slots.append(slot_id)
            return True
        return False

    def get_stats(self) -> dict:
        reply = self._send(ClientCmd.STATS, {})
        if not reply.get("ok"):
            raise RuntimeError(f"STATS failed: {reply.get('error')}")
        return reply

    def _atexit_cleanup(self) -> None:
        if not self._closed:
            try:
                self.medium.unlink()
            except Exception:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shutdown.set()
        try:
            self._market.close(linger=0)
        except zmq.ZMQError:
            pass
        try:
            self._ctx.term()
        except zmq.ZMQError:
            pass
        try:
            self.medium.close()
        finally:
            self.medium.unlink()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *args) -> None:
        self.close()
