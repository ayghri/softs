"""Marketplace broker: routes orders from clients to suppliers by product_id."""

import logging
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
        max_queue_per_product: int = 5000,
    ):
        self.endpoints = endpoints
        self.supplier_timeout = supplier_timeout
        self.client_timeout = client_timeout
        self.max_queue_per_product = max_queue_per_product

        self._lock = threading.RLock()
        self._shutdown = threading.Event()

        self._clients: dict[bytes, ClientInfo] = {}
        self._suppliers: dict[bytes, SupplierInfo] = {}
        self._next_client_id = 0
        self._next_supplier_id = 0

        # Single source of truth
        self._orders: dict[str, Order] = {}

        # Indices
        self._client_orders: dict[bytes, set[str]] = defaultdict(set)
        self._queued: dict[str, deque[str]] = defaultdict(deque)  # product_id → order_ids
        self._available: dict[str, deque[bytes]] = defaultdict(deque)  # product_id → supplier_ids
        self._busy: dict[bytes, str] = {}  # supplier_id → order_id

        self._total_completed = 0

    def _setup_zmq(self) -> None:
        self._ctx = zmq.Context()
        self._frontend = self._bind_router(self.endpoints.frontend)
        self._backend = self._bind_router(self.endpoints.backend)

    def _bind_router(self, endpoint: str) -> zmq.Socket:
        sock = self._ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(endpoint)
        return sock

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

    def _touch(self, registry: dict, identity: bytes) -> None:
        with self._lock:
            info = registry.get(identity)
            if info:
                info.last_seen = time.time()

    # -- Client handlers --

    def _handle_client_hello(self, identity: bytes, payload: dict) -> bytes:
        with self._lock:
            existing = self._clients.get(identity)
            if existing:
                existing.last_seen = time.time()
                return make_reply(True, peer_id=existing.peer_id)
            self._next_client_id += 1
            peer_id = self._next_client_id
            self._clients[identity] = ClientInfo(
                peer_id=peer_id, last_seen=time.time()
            )
            total = len(self._clients)
        logger.info(f"Client {peer_id} connected (total={total})")
        return make_reply(True, peer_id=peer_id)

    def _handle_client_order(self, identity: bytes, payload: dict) -> bytes:
        try:
            req = OrderRequest(**payload)
        except TypeError as e:
            return make_reply(False, error=str(e))
        with self._lock:
            self._touch(self._clients, identity)
            if (
                len(self._queued.get(req.product_id, []))
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
            self._client_orders[identity].add(req.order_id)
            self._queued[req.product_id].append(req.order_id)
        self._try_dispatch()
        return make_reply(True)

    def _handle_client_discard(self, identity: bytes, payload: dict) -> bytes:
        with self._lock:
            order_ids = list(self._client_orders.get(identity, set()))
            for oid in order_ids:
                self._remove_order(oid)
            self._client_orders[identity] = set()
        return make_reply(True, cancelled=len(order_ids))

    def _handle_client_cancel(self, identity: bytes, payload: dict) -> bytes:
        try:
            msg = OrderCancel(**payload)
        except TypeError as e:
            return make_reply(False, error=str(e))
        with self._lock:
            order = self._orders.get(msg.order_id)
            if not order or order.client_id != identity:
                return make_reply(False, error="Order not owned by this client")
            self._remove_order(msg.order_id)
        return make_reply(True)

    # -- Supplier handlers --

    def _handle_supplier_hello(self, identity: bytes, payload: dict) -> bytes:
        product_ids = payload.get("product_ids", [])
        if not product_ids:
            return make_reply(
                False, error="Suppliers must register at least one product_id"
            )
        with self._lock:
            existing = self._suppliers.get(identity)
            if existing:
                existing.product_ids = product_ids
                existing.last_seen = time.time()
                return make_reply(True, peer_id=existing.peer_id)
            self._next_supplier_id += 1
            peer_id = self._next_supplier_id
            self._suppliers[identity] = SupplierInfo(
                peer_id=peer_id,
                product_ids=product_ids,
                last_seen=time.time(),
            )
            total = len(self._suppliers)
        logger.info(
            f"Supplier {peer_id} connected (products={product_ids}, total={total})"
        )
        return make_reply(True, peer_id=peer_id)

    def _handle_supplier_ready(self, identity: bytes, payload: dict) -> None:
        with self._lock:
            self._touch(self._suppliers, identity)
            info = self._suppliers.get(identity)
            if info:
                for pid in info.product_ids:
                    q = self._available[pid]
                    if identity not in q:
                        q.append(identity)
        self._try_dispatch()

    def _handle_supplier_done(self, identity: bytes, payload: dict) -> bytes:
        try:
            msg = OrderDone(**payload)
        except TypeError as e:
            return make_reply(False, error=str(e))
        order_id = msg.order_id
        success = msg.success
        with self._lock:
            self._touch(self._suppliers, identity)
            self._busy.pop(identity, None)
            order = self._orders.pop(order_id, None)
            if not order:
                return make_reply(False, error="Unknown order")
            self._client_orders.get(order.client_id, set()).discard(order_id)
            if success:
                self._total_completed += 1
            else:
                order.state = OrderState.QUEUED
                order.supplier_id = b""
                order.dispatched_at = 0.0
                self._orders[order_id] = order
                self._client_orders[order.client_id].add(order_id)
                self._queued[order.product_id].append(order_id)
        if success:
            self._frontend.send_multipart(
                [
                    order.client_id,
                    ClientCmd.FULFILLED,
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

    def _try_dispatch(self) -> None:
        with self._lock:
            for product_id, queue in self._queued.items():
                avail = self._available.get(product_id)
                if not avail:
                    continue
                while queue and avail:
                    order_id = queue.popleft()
                    order = self._orders.get(order_id)
                    if not order:
                        continue  # cancelled while queued
                    supplier = avail.popleft()
                    order.state = OrderState.DISPATCHED
                    order.supplier_id = supplier
                    order.dispatched_at = time.time()
                    self._busy[supplier] = order_id
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
        self._client_orders.get(order.client_id, set()).discard(order_id)
        if order.state == OrderState.DISPATCHED and order.supplier_id:
            self._busy.pop(order.supplier_id, None)

    # -- Disconnect handling --

    def _handle_supplier_disconnect(self, supplier_id: bytes) -> None:
        with self._lock:
            info = self._suppliers.pop(supplier_id, None)
            if info:
                for pid in info.product_ids:
                    q = self._available.get(pid)
                    if q:
                        try:
                            q.remove(supplier_id)
                        except ValueError:
                            pass
            order_id = self._busy.pop(supplier_id, None)
            if order_id:
                order = self._orders.get(order_id)
                if order:
                    order.state = OrderState.QUEUED
                    order.supplier_id = b""
                    order.dispatched_at = 0.0
                    self._queued[order.product_id].append(order_id)
        if info:
            logger.warning("Supplier disconnected")
        self._try_dispatch()

    def _handle_client_disconnect(self, client_id: bytes) -> None:
        with self._lock:
            for oid in list(self._client_orders.pop(client_id, set())):
                self._remove_order(oid)
            self._clients.pop(client_id, None)

    def _requeue_stale_work(self, timeout: float = 30.0) -> int:
        now = time.time()
        with self._lock:
            stale_suppliers: set[bytes] = set()
            for supplier_id, order_id in list(self._busy.items()):
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
        now = time.time()
        with self._lock:
            dead_s = [
                i
                for i, info in self._suppliers.items()
                if info.last_seen > 0
                and (now - info.last_seen) > self.supplier_timeout
            ]
            dead_c = [
                i
                for i, info in self._clients.items()
                if info.last_seen > 0
                and (now - info.last_seen) > self.client_timeout
            ]
            for k in [k for k, v in self._queued.items() if not v]:
                del self._queued[k]
            for k in [k for k, v in self._available.items() if not v]:
                del self._available[k]
        for i in dead_s:
            self._handle_supplier_disconnect(i)
        for i in dead_c:
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
            self._poll_thread.join(timeout=1.0)
        self._teardown_zmq()

    def get_stats(self) -> BrokerStats:
        with self._lock:
            all_products: set[str] = set()
            for info in self._suppliers.values():
                all_products.update(info.product_ids)
            return BrokerStats(
                pending_orders=len(self._orders),
                available_suppliers=sum(
                    len(q) for q in self._available.values()
                ),
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
