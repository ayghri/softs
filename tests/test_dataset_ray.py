"""Tests for Ray-based dataset integration."""

import pytest
import ray
import torch
from torch.utils.data import DataLoader

from softlabels import Supplier, SupplierPool, SoftIterableDataset, SoftDataLoader


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(num_cpus=4, ignore_reinit_error=True)
    yield
    ray.shutdown()


def simple_gen(product_id: str) -> dict[str, torch.Tensor]:
    return {"x": torch.randn(3, 8, 8), "y": torch.randn(4)}


@pytest.fixture
def pool():
    return SupplierPool([Supplier(simple_gen)], prefetch=2)


class TestSoftIterableDataset:
    def test_yields_tensors(self, pool):
        dataset = SoftIterableDataset(pool, model_id="test")
        for i, sample in enumerate(dataset):
            assert sample["x"].shape == (3, 8, 8)
            assert sample["y"].shape == (4,)
            if i >= 4:
                break

    def test_with_dataloader(self, pool):
        dataset = SoftIterableDataset(pool, model_id="test")
        loader = DataLoader(dataset, batch_size=2)
        for i, batch in enumerate(loader):
            assert batch["x"].shape == (2, 3, 8, 8)
            assert batch["y"].shape == (2, 4)
            if i >= 2:
                break

    def test_model_switching(self):
        class SwitchableGen:
            def __init__(self):
                self._val = 1.0

            def __call__(self, product_id):
                return {"x": torch.full((1,), self._val)}

            def set_model(self, model_id):
                self._val = float(model_id)

        pool = SupplierPool([Supplier(SwitchableGen)], prefetch=1)
        dataset = SoftIterableDataset(pool, model_id="1.0")

        it = iter(dataset)
        assert next(it)["x"].item() == 1.0

        dataset.set_model("2.0")
        for _ in range(3):
            sample = next(it)
        assert sample["x"].item() == 2.0


class TestSoftDataLoader:
    def test_basic(self, pool):
        loader = SoftDataLoader(pool, model_id="test", batch_size=3)
        for i, batch in enumerate(loader):
            assert batch["x"].shape == (3, 3, 8, 8)
            if i >= 2:
                break

    def test_model_property(self, pool):
        loader = SoftDataLoader(pool, model_id="test")
        assert loader.model_id == "test"
        loader.set_model("other")
        assert loader.model_id == "other"
