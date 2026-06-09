"""Marketplace broker: routes orders to suppliers by matching product_id patterns.

An order carries a concrete ``product_id`` string; a supplier registers one or
more **regex patterns** (its ``product_ids``) describing what it can serve. The
broker dispatches an order to a ready supplier whose pattern ``fullmatch``-es the
order's ``product_id`` - so a supplier declares a pattern instead of enumerating
every allowed id. A metachar-free pattern behaves as an exact match.
"""

import logging
import re
import threading
import time
from collections import defaultdict, deque

import zmq

from .protocol import EndpointConfig
from .protocol import (
    ClientCmd,
    SupplierCmd,
    Order,
    OrderState,
    ClientInfo,
    SupplierInfo,
    BrokerStats,
    OrderRequest,
    OrderCancel,
    OrderDone,
    encode_payload,
    make_reply,
    parse_request,
)

logger = logging.getLogger(__name__)


class Broker:

    def __init__(
        self,
        endpoints: EndpointConfig,
        supplier_timeout: float = 60.0,
        client_timeout: float = 120.0,
        max_queue_per_product: int = 16,
        max_order_attempts: int = 5,
    ):
        self.endpoints = endpoints
        self.supplier_timeout = supplier_timeout
        self.client_timeout = client_timeout
        self.max_queue_per_product = max_queue_per_product
        self.max_order_attempts = max_order_attempts

        self._lock = threading.RLock()
        self._shutdown = threading.Event()

        self._clients: dict[bytes, ClientInfo] = {}
        self._suppliers: dict[bytes, SupplierInfo] = {}
        self._next_client_id = 0
        self._next_supplier_id = 0

        # Single source of truth
        self._orders: dict[str, Order] = {}

        # mappings, s:supplier, c:client, p:product, o:order
        self._cid_to_oid: dict[bytes, set[str]] = defaultdict(set)
        # product_id -> order_ids
        self._pid_to_oid: dict[str, deque[str]] = defaultdict(deque)
        # ready suppliers (FIFO); each can serve any product_id its patterns match
        self._ready: deque[bytes] = deque()
        # supplier_id -> compiled regex patterns it serves
        self._patterns: dict[bytes, list[re.Pattern]] = {}
        # supplier_id -> order_id
        self._sid_to_oid: dict[bytes, str] = {}

        self._total_completed = 0

    def _bind_router(self, endpoint: str) -> zmq.Socket:
        sock = self._ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(endpoint)
        return sock

    def _setup_zmq(self) -> None:
        self._ctx = zmq.Context()
        self._frontend = self._bind_router(self.endpoints.frontend)
        self._backend = self._bind_router(self.endpoints.backend)

    def _teardown_zmq(self) -> None:
        for sock in (self._frontend, self._backend):
            if sock:
                try:
                    sock.close(linger=0)
                except zmq.ZMQError:
                    pass
        try:
            self._ctx.term()
        except zmq.ZMQError:
            pass

    def _touch(self, registry: dict, identity: bytes) -> bool:
        with self._lock:
            info = registry.get(identity)
            if info:
                info.last_seen = time.time()
                return True
        return False

    def _add_party(self, registry, identity, info: dict):
        registry[identity] = ClientInfo(peer_id=identity, **info)

        # )

    # -- Client handlers --

    def _handle_client_hello(self, identity: bytes, payload: dict) -> bytes:
        with self._lock:
            if self._touch(self._clients, identity):
                return make_reply(True, peer_id=self._clients[identity].peer_id)
            self._next_client_id += 1
            self._add_party(
                self._clients, self._next_client_id, {"last_seen": time.time()}
            )

            total = len(self._clients)
        logger.info(f"Client {self._next_client_id} connected (total={total})")
        return make_reply(True, peer_id=self._next_client_id)

    def _handle_client_order(self, identity: bytes, payload: dict) -> bytes:
        try:
            req = OrderRequest(**payload)
        except TypeError as e:
            return make_reply(False, error=str(e))
        with self._lock:
            self._touch(self._clients, identity)
            if (
                len(self._pid_to_oid.get(req.product_id, []))
                >= self.max_queue_per_product
            ):
                return make_reply(False, error="Queue full for product_id")
            order = Order(
                order_id=req.order_id,
                client_id=identity,
                product_id=req.product_id,
                address=req.address,
                offset=req.offset,
            )
            self._orders[req.order_id] = order
            self._cid_to_oid[identity].add(req.order_id)
            self._pid_to_oid[req.product_id].append(req.order_id)
        self._try_dispatch()
        return make_reply(True)

    def _handle_client_discard(self, identity: bytes, payload: dict) -> bytes:
        # Only cancel orders that are still QUEUED (no supplier has touched their
        # slot). DISPATCHED orders are left running so they FULFILL normally -
        # the client quarantines those slots until then, preventing a supplier
        # write from landing in a reused slot. See Client.discard().
        with self._lock:
            cancelled: list[str] = []
            for order_id in list(self._cid_to_oid.get(identity, set())):
                order = self._orders.get(order_id)
                if order is not None and order.status == OrderState.QUEUED:
                    self._remove_order(order_id)
                    cancelled.append(order_id)
        return make_reply(
            True, cancelled=len(cancelled), cancelled_ids=cancelled
        )

    def _handle_client_cancel(self, identity: bytes, payload: dict) -> bytes:
        try:
            msg = OrderCancel(**payload)
        except TypeError as e:
            return make_reply(False, error=str(e))
        with self._lock:
            order = self._orders.get(msg.order_id)
            if not order or order.client_id != identity:
                return make_reply(False, error="Order not owned by this client")
            # Only QUEUED orders can be safely cancelled; a DISPATCHED order is
            # already being written by a supplier, so let it FULFILL normally.
            if order.status != OrderState.QUEUED:
                return make_reply(False, error="Order already dispatched")
            self._remove_order(msg.order_id)
        return make_reply(True)

    # -- Supplier handlers --

    def _handle_supplier_hello(self, identity: bytes, payload: dict) -> bytes:
        product_ids = payload.get("product_ids", [])
        if not product_ids:
            return make_reply(
                False, error="Suppliers must register at least one product_id pattern"
            )
        try:
            patterns = [re.compile(p) for p in product_ids]
        except re.error as e:
            return make_reply(False, error=f"Invalid product_id pattern: {e}")
        with self._lock:
            existing = self._suppliers.get(identity)
            if existing:
                existing.product_ids = product_ids
                existing.last_seen = time.time()
                self._patterns[identity] = patterns
                return make_reply(True, peer_id=existing.peer_id)
            self._next_supplier_id += 1
            peer_id = self._next_supplier_id
            self._suppliers[identity] = SupplierInfo(
                peer_id=peer_id,
                product_ids=product_ids,
                last_seen=time.time(),
            )
            self._patterns[identity] = patterns
            total = len(self._suppliers)
        logger.info(
            f"Supplier {peer_id} connected (patterns={product_ids}, total={total})"
        )
        return make_reply(True, peer_id=peer_id)

    def _handle_supplier_ready(self, identity: bytes, payload: dict) -> None:
        with self._lock:
            if self._touch(self._suppliers, identity):
                if identity not in self._ready:
                    self._ready.append(identity)
        self._try_dispatch()

    def _handle_supplier_done(self, identity: bytes, payload: dict) -> bytes:
        try:
            msg = OrderDone(**payload)
        except TypeError as e:
            return make_reply(False, error=str(e))
        order_id = msg.order_id
        success = msg.success
        gave_up = False
        with self._lock:
            self._touch(self._suppliers, identity)
            self._sid_to_oid.pop(identity, None)
            order = self._orders.pop(order_id, None)
            if not order:
                return make_reply(False, error="Unknown order")
            self._cid_to_oid.get(order.client_id, set()).discard(order_id)
            if success:
                self._total_completed += 1
            else:
                order.attempts += 1
                if order.attempts >= self.max_order_attempts:
                    # Permanent failure: drop the order (already removed from all
                    # indices above) and tell the client to free its slot, so a
                    # broken supplier/medium can't loop forever or leak slots.
                    gave_up = True
                    logger.error(
                        "Order %s (product=%s) failed %d times; giving up. "
                        "Supplier writes are failing - check the medium/offset.",
                        order_id,
                        order.product_id,
                        order.attempts,
                    )
                else:
                    if order.attempts == 1:
                        logger.warning(
                            "Order %s (product=%s) failed; requeueing "
                            "(attempt %d/%d)",
                            order_id,
                            order.product_id,
                            order.attempts,
                            self.max_order_attempts,
                        )
                    order.status = OrderState.QUEUED
                    order.supplier_id = b""
                    order.dispatched_at = 0.0
                    self._orders[order_id] = order
                    self._cid_to_oid[order.client_id].add(order_id)
                    self._pid_to_oid[order.product_id].append(order_id)
        if success:
            self._frontend.send_multipart(
                [
                    order.client_id,
                    ClientCmd.FULFILLED,
                    encode_payload({"order_id": order_id}),
                ]
            )
        elif gave_up:
            self._frontend.send_multipart(
                [
                    order.client_id,
                    ClientCmd.FAILED,
                    encode_payload({"order_id": order_id}),
                ]
            )
        else:
            self._try_dispatch()
        return make_reply(True)

    def _handle_supplier_goodbye(self, identity: bytes, payload: dict) -> bytes:
        self._handle_supplier_disconnect(identity)
        return make_reply(True)

    # -- Shared --

    def _handle_stats(self, identity: bytes, payload: dict) -> bytes:
        return make_reply(True, **vars(self.get_stats()))

    # -- Dispatch --

    def _supplier_matches(self, supplier_id: bytes, product_id: str) -> bool:
        return any(
            p.fullmatch(product_id) for p in self._patterns.get(supplier_id, ())
        )

    def _take_ready(self, product_id: str) -> bytes | None:
        """Pop the first ready supplier whose pattern matches product_id."""
        for idx, sid in enumerate(self._ready):
            if self._supplier_matches(sid, product_id):
                del self._ready[idx]
                return sid
        return None

    def _try_dispatch(self) -> None:
        with self._lock:
            for product_id, queue in self._pid_to_oid.items():
                while queue and self._ready:
                    order_id = queue.popleft()
                    order = self._orders.get(order_id)
                    if not order:
                        continue  # cancelled while queued; don't consume a supplier
                    supplier = self._take_ready(product_id)
                    if supplier is None:
                        # No ready supplier serves this product_id; put it back.
                        queue.appendleft(order_id)
                        break
                    order.status = OrderState.DISPATCHED
                    order.supplier_id = supplier
                    order.dispatched_at = time.time()
                    self._sid_to_oid[supplier] = order_id
                    self._backend.send_multipart(
                        [
                            supplier,
                            SupplierCmd.WORK,
                            encode_payload(
                                {
                                    "order_id": order.order_id,
                                    "product_id": order.product_id,
                                    "address": order.address,
                                    "offset": order.offset,
                                }
                            ),
                        ]
                    )

    def _remove_order(self, order_id: str) -> None:
        """Remove order from all indices. Must hold _lock."""
        order = self._orders.pop(order_id, None)
        if not order:
            return
        self._cid_to_oid.get(order.client_id, set()).discard(order_id)
        if order.status == OrderState.DISPATCHED and order.supplier_id:
            self._sid_to_oid.pop(order.supplier_id, None)

    # -- Disconnect handling --

    def _handle_supplier_disconnect(self, supplier_id: bytes) -> None:
        with self._lock:
            info = self._suppliers.pop(supplier_id, None)
            self._patterns.pop(supplier_id, None)
            try:
                self._ready.remove(supplier_id)
            except ValueError:
                pass
            order_id = self._sid_to_oid.pop(supplier_id, None)
            if order_id:
                order = self._orders.get(order_id)
                if order:
                    order.status = OrderState.QUEUED
                    order.supplier_id = b""
                    order.dispatched_at = 0.0
                    self._pid_to_oid[order.product_id].append(order_id)
        if info:
            logger.warning("Supplier disconnected")
        self._try_dispatch()

    def _handle_client_disconnect(self, client_id: bytes) -> None:
        with self._lock:
            for oid in list(self._cid_to_oid.pop(client_id, set())):
                self._remove_order(oid)
            self._clients.pop(client_id, None)

    def _requeue_stale_work(self, timeout: float = 30.0) -> int:
        now = time.time()
        with self._lock:
            stale_suppliers: set[bytes] = set()
            for supplier_id, order_id in list(self._sid_to_oid.items()):
                order = self._orders.get(order_id)
                if (
                    order
                    and order.dispatched_at > 0
                    and (now - order.dispatched_at) > timeout
                ):
                    stale_suppliers.add(supplier_id)
        for sid in stale_suppliers:
            self._handle_supplier_disconnect(sid)
        return len(stale_suppliers)

    def _check_liveness(self) -> None:

        with self._lock:
            now = time.time()
            dead_suppliers = [
                i
                for i, info in self._suppliers.items()
                if (now - info.last_seen) > self.supplier_timeout
            ]
            dead_clients = [
                i
                for i, info in self._clients.items()
                if info.last_seen > 0
                and (now - info.last_seen) > self.client_timeout
            ]

            for i in dead_suppliers:
                self._handle_supplier_disconnect(i)
            for i in dead_clients:
                self._handle_client_disconnect(i)

    # -- Socket processing --

    def _process_socket(self, sock: zmq.Socket, handlers: dict) -> None:
        frames = sock.recv_multipart()
        if len(frames) < 2:
            return
        identity, msg_frames = frames[0], frames[1:]
        try:
            cmd, payload = parse_request(msg_frames)
        except Exception as e:
            sock.send_multipart([identity, make_reply(False, error=str(e))])
            return
        handler = handlers.get(cmd)
        if not handler:
            sock.send_multipart(
                [identity, make_reply(False, error=f"Unknown command: {cmd}")]
            )
            return
        result = handler(identity, payload)
        if result is not None:
            sock.send_multipart([identity, result])

    def _poll_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._frontend, zmq.POLLIN)
        poller.register(self._backend, zmq.POLLIN)

        frontend_handlers = {
            ClientCmd.HELLO: self._handle_client_hello,
            ClientCmd.ORDER: self._handle_client_order,
            ClientCmd.DISCARD: self._handle_client_discard,
            ClientCmd.CANCEL: self._handle_client_cancel,
            ClientCmd.STATS: self._handle_stats,
        }
        backend_handlers = {
            SupplierCmd.HELLO: self._handle_supplier_hello,
            SupplierCmd.READY: self._handle_supplier_ready,
            SupplierCmd.DONE: self._handle_supplier_done,
            SupplierCmd.GOODBYE: self._handle_supplier_goodbye,
        }

        last_stale = last_liveness = time.time()
        while not self._shutdown.is_set():
            try:
                socks = dict(poller.poll(100))
                if self._frontend in socks:
                    self._process_socket(self._frontend, frontend_handlers)
                if self._backend in socks:
                    self._process_socket(self._backend, backend_handlers)
                now = time.time()
                if now - last_stale > 5.0:
                    requeued = self._requeue_stale_work(timeout=10.0)
                    if requeued > 0:
                        logger.warning(f"Re-queued {requeued} stale orders")
                    last_stale = now
                if now - last_liveness > 10.0:
                    self._check_liveness()
                    last_liveness = now
            except zmq.ZMQError as e:
                if not self._shutdown.is_set():
                    logger.error(f"ZMQ error: {e}")
                break

    def start(self) -> None:
        self._setup_zmq()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True
        )
        self._poll_thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        if hasattr(self, "_poll_thread"):
            self._poll_thread.join(timeout=10.0)
        self._teardown_zmq()

    def get_stats(self) -> BrokerStats:
        with self._lock:
            all_products: set[str] = set()
            for info in self._suppliers.values():
                all_products.update(info.product_ids)
            return BrokerStats(
                pending_orders=len(self._orders),
                available_suppliers=len(self._ready),
                connected_clients=len(self._clients),
                connected_suppliers=len(self._suppliers),
                total_completed=self._total_completed,
                product_ids=list(all_products),
            )

    def __enter__(self) -> "Broker":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
