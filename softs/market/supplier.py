"""Supplier process — generates samples and writes to transfer medium."""

import logging
import threading
from typing import Callable

import zmq

from .mediums import Medium
from .protocol import SupplierCmd, OrderWork, decode_payload, make_request

logger = logging.getLogger(__name__)


class Supplier:
    def __init__(
        self,
        generator_fn: Callable[[str], bytes],
        product_ids: list[str],
        endpoint: str,
        medium_cls: type[Medium],
        slot_size: int,
        send_timeout_ms: int = 10000,
    ):
        self.generator_fn = generator_fn
        self.product_ids = product_ids
        self.slot_size = slot_size
        self.endpoint = endpoint
        self._medium_cls = medium_cls
        self.send_timeout_ms = send_timeout_ms
        self._peer_id: int | None = None

        self._shutdown = threading.Event()
        self._writer_cache: dict[str, Medium] = {}
        self._main_thread: threading.Thread | None = None

        self._ctx = zmq.Context()
        self._ctx.linger = 0
        self._backend = self._ctx.socket(zmq.DEALER)
        self._backend.setsockopt(zmq.LINGER, 0)
        self._backend.connect(self.endpoint)

    def _close_writer_cache(self) -> None:
        for w in self._writer_cache.values():
            try:
                w.close()
            except Exception:
                logger.debug("Failed to close medium", exc_info=True)
        self._writer_cache.clear()

    def _get_writer(self, address: str) -> Medium | None:
        if address in self._writer_cache:
            return self._writer_cache[address]
        try:
            medium = self._medium_cls.attach(address)
            self._writer_cache[address] = medium
            return medium
        except Exception:
            logger.warning(
                f"Failed to attach to medium '{address}'", exc_info=True
            )
            return None

    def _send(self, cmd: bytes, payload: dict) -> dict:
        self._backend.send_multipart(make_request(cmd, payload))
        if self._backend.poll(self.send_timeout_ms):
            return decode_payload(self._backend.recv_multipart()[-1])
        return {"ok": False, "error": "timeout"}

    def _hello(self) -> dict:
        reply = self._send(
            SupplierCmd.HELLO,
            {"product_ids": self.product_ids},
        )
        if not reply.get("ok"):
            raise RuntimeError(f"HELLO failed: {reply.get('error')}")
        self._peer_id = reply["peer_id"]
        return reply

    def _send_ready(self):
        self._backend.send_multipart(make_request(SupplierCmd.READY, {}))

    def _send_done(self, order_id: str, success: bool = True):
        return self._send(
            SupplierCmd.DONE, {"order_id": order_id, "success": success}
        )

    def _process_work(self, payload: dict) -> None:
        try:
            work = OrderWork(**payload)
        except TypeError as e:
            logger.error(f"Invalid WORK payload: {e}")
            return
        try:
            data = self.generator_fn(work.product_id)
            writer = self._get_writer(work.address)
            if writer is None or not writer.write(work.offset, data):
                self._writer_cache.pop(work.address, None)
                self._send_done(work.order_id, success=False)
                return
            self._send_done(work.order_id, success=True)
        except Exception as e:
            logger.error(f"Generator error: {e}")
            self._send_done(work.order_id, success=False)

    def _main_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._backend, zmq.POLLIN)
        self._send_ready()

        while not self._shutdown.is_set():
            try:
                socks = dict(poller.poll(50))
                if self._backend in socks:
                    frames = self._backend.recv_multipart()
                    if len(frames) >= 2:
                        cmd = frames[0]
                        payload = decode_payload(frames[1]) if frames[1] else {}
                        if cmd == SupplierCmd.WORK:
                            self._process_work(payload)
                            self._send_ready()
                        elif cmd == SupplierCmd.TERMINATE:
                            self.stop()
            except zmq.ZMQError as e:
                if not self._shutdown.is_set():
                    logger.error(f"ZMQ error: {e}")
                break

    def start(self) -> None:
        self._hello()
        self._main_thread = threading.Thread(
            target=self._main_loop, daemon=True
        )
        self._main_thread.start()

    def stop(self) -> None:
        try:
            self._backend.send_multipart(make_request(SupplierCmd.GOODBYE, {}))
        except zmq.ZMQError:
            pass
        self._shutdown.set()
        if self._main_thread is not None:
            self._main_thread.join(timeout=2.0)
        self._close_writer_cache()
        try:
            self._backend.close(linger=0)
        except zmq.ZMQError:
            pass
        try:
            self._ctx.term()
        except zmq.ZMQError:
            pass

    def __enter__(self) -> "Supplier":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
