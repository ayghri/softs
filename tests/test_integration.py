"""Integration tests for the marketplace architecture."""
import time

import pytest
import torch
from torch.utils.data import DataLoader

from softs import (
    ShmMedium,
    Broker, Supplier, Client, SoftIterableDataset,
    Batch, make_collate_fn, BatchConfig, TensorSpec, EndpointConfig,
)

ENDPOINTS = EndpointConfig(
    frontend="ipc:///tmp/sl_integ_fe.sock",
    backend="ipc:///tmp/sl_integ_be.sock",
)
X_SHAPE = (3, 8, 8)
Y_SHAPE = (4,)


@pytest.fixture
def config():
    return BatchConfig([TensorSpec("x", X_SHAPE, "float32"), TensorSpec("y", Y_SHAPE, "float32")])


def make_gen(config):
    def gen(model_id):
        return config.encode(x=torch.randn(*X_SHAPE), y=torch.randn(*Y_SHAPE))
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


class TestEndToEnd:
    def test_full_pipeline(self, supplier, client, config):
        for i in range(5):
            slot = client.request_sample("test", timeout_ms=2000)
            assert slot is not None, f"Failed at sample {i}"
            tensors = config.decode(client.medium.read(slot))
            assert tensors["x"].shape == X_SHAPE
            assert tensors["y"].shape == Y_SHAPE
            client.release_slot(slot)


class TestDataset:
    def test_dataset_yields_tensors(self, supplier, config):
        dataset = SoftIterableDataset(
            model_id="test", num_slots=4,
            batch_config=config, endpoint=ENDPOINTS.frontend,
            medium_cls=ShmMedium,
        )
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
        for i, batch in enumerate(dataloader):
            assert batch["x"].shape == (2, *X_SHAPE)
            assert batch["y"].shape == (2, *Y_SHAPE)
            if i >= 2:
                break


class TestSlotManagement:
    def test_slots_recycled(self, supplier, client):
        initial_free = len(client._free_slots)
        slots = []
        for _ in range(initial_free):
            slot = client.request_sample("test", timeout_ms=2000)
            if slot is not None:
                slots.append(slot)
        for slot in slots:
            client.release_slot(slot)
        assert len(client._free_slots) == initial_free

    def test_discard_clears_pending(self, supplier, client):
        for _ in range(3):
            client.request_slot("test")
        time.sleep(0.3)
        client.discard()
        assert len(client._pending_slots) == 0
