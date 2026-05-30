"""Tests for the marketplace broker."""
import time

import pytest
import torch

from softs import Broker, Supplier, Client, ShmMedium, BatchConfig, TensorSpec, EndpointConfig

ENDPOINTS = EndpointConfig(
    frontend="ipc:///tmp/sl_test_fe.sock",
    backend="ipc:///tmp/sl_test_be.sock",
)


@pytest.fixture
def config():
    return BatchConfig([TensorSpec("x", (3, 8, 8), "float32"), TensorSpec("y", (4,), "float32")])


def make_gen(config):
    def gen(model_id):
        return config.encode(x=torch.randn(3, 8, 8), y=torch.randn(4))
    return gen


@pytest.fixture
def broker():
    b = Broker(endpoints=ENDPOINTS)
    b.start()
    time.sleep(0.1)
    yield b
    b.stop()


@pytest.fixture
def supplier(broker, config):
    w = Supplier(medium_cls=ShmMedium, generator_fn=make_gen(config), product_ids=["test"],
               slot_size=config.nbytes(), endpoint=ENDPOINTS.backend)
    w.start()
    time.sleep(0.1)
    yield w
    w.stop()


@pytest.fixture
def client(broker, config):
    c = Client(medium_cls=ShmMedium, num_slots=8, slot_size=config.nbytes(), endpoint=ENDPOINTS.frontend)
    c.hello()
    yield c
    c.close()


class TestBasic:
    def test_broker_starts_and_stops(self):
        ep = EndpointConfig(
            frontend="ipc:///tmp/sl_test_basic_fe.sock",
            backend="ipc:///tmp/sl_test_basic_be.sock",
        )
        b = Broker(endpoints=ep)
        b.start()
        time.sleep(0.1)
        b.stop()

    def test_client_hello(self, broker, config):
        c = Client(medium_cls=ShmMedium, num_slots=4, slot_size=config.nbytes(), endpoint=ENDPOINTS.frontend)
        try:
            reply = c.hello()
            assert reply["ok"]
        finally:
            c.close()

    def test_supplier_hello(self, broker, config):
        w = Supplier(medium_cls=ShmMedium, generator_fn=make_gen(config), product_ids=["m1"],
                   slot_size=config.nbytes(), endpoint=ENDPOINTS.backend)
        try:
            w.start()
        finally:
            w.stop()


class TestRouting:
    def test_request_dispatched(self, supplier, client):
        slot = client.request_sample("test", timeout_ms=2000)
        assert slot is not None
        client.release_slot(slot)

    def test_data_written(self, supplier, client, config):
        slot = client.request_sample("test", timeout_ms=2000)
        assert slot is not None
        tensors = config.decode(client.medium.read(slot))
        assert tensors["x"].shape == (3, 8, 8)
        assert tensors["y"].shape == (4,)
        client.release_slot(slot)

    def test_multiple_requests(self, supplier, client, config):
        for _ in range(5):
            slot = client.request_sample("test", timeout_ms=2000)
            assert slot is not None
            config.decode(client.medium.read(slot))
            client.release_slot(slot)

    def test_model_routing(self, broker, config):
        def gen_a(mid): return config.encode(x=torch.ones(3, 8, 8), y=torch.ones(4))
        def gen_b(mid): return config.encode(x=torch.zeros(3, 8, 8), y=torch.zeros(4))

        wa = Supplier(medium_cls=ShmMedium, generator_fn=gen_a, product_ids=["a"], slot_size=config.nbytes(), endpoint=ENDPOINTS.backend)
        wb = Supplier(medium_cls=ShmMedium, generator_fn=gen_b, product_ids=["b"], slot_size=config.nbytes(), endpoint=ENDPOINTS.backend)
        wa.start(); wb.start()
        time.sleep(0.2)

        c = Client(medium_cls=ShmMedium, num_slots=8, slot_size=config.nbytes(), endpoint=ENDPOINTS.frontend)
        c.hello()

        try:
            slot = c.request_sample("a", timeout_ms=2000)
            assert slot is not None
            assert config.decode(c.medium.read(slot))["x"].sum() > 0
            c.release_slot(slot)

            slot = c.request_sample("b", timeout_ms=2000)
            assert slot is not None
            assert config.decode(c.medium.read(slot))["x"].sum() == 0
            c.release_slot(slot)
        finally:
            c.close(); wa.stop(); wb.stop()


class TestDiscard:
    def test_discard_all(self, supplier, client):
        client.request_slot("test")
        client.request_slot("test")
        time.sleep(0.3)
        cancelled = client.discard()
        assert cancelled >= 0
        assert len(client._pending_slots) == 0

    def test_cancel_specific(self, broker, config):
        # Use slow supplier so request stays pending
        def slow_gen(mid):
            time.sleep(10)
            return config.encode(x=torch.randn(3, 8, 8), y=torch.randn(4))

        w = Supplier(medium_cls=ShmMedium, generator_fn=slow_gen, product_ids=["slow"],
                   slot_size=config.nbytes(), endpoint=ENDPOINTS.backend)
        w.start()
        time.sleep(0.2)

        c = Client(medium_cls=ShmMedium, num_slots=8, slot_size=config.nbytes(), endpoint=ENDPOINTS.frontend)
        c.hello()

        try:
            token = c.request_slot("slow")
            assert token is not None
            time.sleep(0.2)
            ok = c.cancel(token)
            assert ok
            assert token not in c._pending_slots
        finally:
            c.close(); w.stop()


class TestMultipleSuppliers:
    def test_multiple_suppliers_same_model(self, broker, config):
        suppliers = []
        for _ in range(3):
            w = Supplier(medium_cls=ShmMedium, generator_fn=make_gen(config), product_ids=["shared"],
                       slot_size=config.nbytes(), endpoint=ENDPOINTS.backend)
            w.start()
            suppliers.append(w)
        time.sleep(0.2)

        c = Client(medium_cls=ShmMedium, num_slots=8, slot_size=config.nbytes(), endpoint=ENDPOINTS.frontend)
        c.hello()

        try:
            for _ in range(10):
                slot = c.request_sample("shared", timeout_ms=2000)
                assert slot is not None
                c.release_slot(slot)
        finally:
            c.close()
            for w in suppliers:
                w.stop()


class TestMultipleClients:
    def test_multiple_clients(self, supplier, config):
        clients = []
        for _ in range(3):
            c = Client(medium_cls=ShmMedium, num_slots=4, slot_size=config.nbytes(), endpoint=ENDPOINTS.frontend)
            c.hello()
            clients.append(c)

        try:
            for c in clients:
                slot = c.request_sample("test", timeout_ms=2000)
                assert slot is not None
                c.release_slot(slot)
        finally:
            for c in clients:
                c.close()


class TestStats:
    def test_stats(self, supplier, client):
        stats = client.get_stats()
        assert stats["ok"]
        assert stats["connected_suppliers"] >= 1
        assert stats["connected_clients"] >= 1
        assert "product_ids" in stats


class TestSupplierDisconnect:
    def test_supplier_disconnect_requeues(self, broker, client, config):
        def slow_gen(mid):
            time.sleep(10)
            return config.encode(x=torch.randn(3, 8, 8), y=torch.randn(4))

        w = Supplier(medium_cls=ShmMedium, generator_fn=slow_gen, product_ids=["slow"],
                   slot_size=config.nbytes(), endpoint=ENDPOINTS.backend)
        w.start()
        time.sleep(0.2)

        token = client.request_slot("slow")
        assert token is not None
        time.sleep(0.3)

        with broker._lock:
            supplier_ids = list(broker._suppliers.keys())
        for wid in supplier_ids:
            broker._handle_supplier_disconnect(wid)
        w.stop()

        w2 = Supplier(medium_cls=ShmMedium, generator_fn=make_gen(config), product_ids=["slow"],
                    slot_size=config.nbytes(), endpoint=ENDPOINTS.backend)
        w2.start()
        time.sleep(0.5)

        client.poll_completions(timeout_ms=2000)
        slot = client.get_completed()
        w2.stop()
