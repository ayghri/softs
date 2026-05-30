"""Tests for Ray-based supplier and pool."""

import pytest
import ray
import torch

from softlabels import Supplier, SupplierPool


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(num_cpus=4, ignore_reinit_error=True)
    yield
    ray.shutdown()


def simple_gen(product_id: str) -> dict[str, torch.Tensor]:
    return {"x": torch.randn(3, 8, 8), "y": torch.randn(4)}


class StatefulGen:
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, product_id: str) -> dict[str, torch.Tensor]:
        return {"x": torch.ones(4) * self.scale}


@pytest.fixture
def supplier():
    return Supplier(simple_gen)


@pytest.fixture
def pool(supplier):
    return SupplierPool([supplier], prefetch=2)


class TestSupplier:
    def test_generate_returns_ref(self, supplier):
        ref = supplier.generate("test")
        assert isinstance(ref, ray.ObjectRef)

    def test_generate_returns_tensors(self, supplier):
        result = ray.get(supplier.generate("test"))
        assert "x" in result
        assert "y" in result
        assert result["x"].shape == (3, 8, 8)
        assert result["y"].shape == (4,)

    def test_class_generator(self):
        s = Supplier(StatefulGen, 3.0)
        result = ray.get(s.generate("test"))
        assert result["x"].sum() == 12.0

    def test_multiple_requests(self, supplier):
        refs = [supplier.generate("test") for _ in range(5)]
        results = ray.get(refs)
        assert len(results) == 5
        for r in results:
            assert r["x"].shape == (3, 8, 8)


class TestSupplierPool:
    def test_get_returns_tensors(self, pool):
        result = pool.get("test")
        assert result is not None
        assert "x" in result
        assert result["x"].shape == (3, 8, 8)

    def test_multiple_gets(self, pool):
        for _ in range(10):
            result = pool.get("test")
            assert result is not None
            assert "x" in result

    def test_prefetch_fills_pipeline(self, pool):
        pool.get("test")
        assert len(pool._pending) >= 1

    def test_discard(self, pool):
        pool.get("test")
        n = pool.discard()
        assert n >= 0
        assert len(pool._pending) == 0

    def test_multiple_suppliers(self):
        suppliers = [Supplier(simple_gen) for _ in range(3)]
        pool = SupplierPool(suppliers, prefetch=6)
        for _ in range(10):
            result = pool.get("test")
            assert result is not None

    def test_timeout_returns_none(self):
        def slow_gen(pid):
            import time
            time.sleep(10)
            return {"x": torch.zeros(1)}

        s = Supplier(slow_gen)
        pool = SupplierPool([s], prefetch=1)
        result = pool.get("test", timeout=0.01)
        assert result is None
        pool.discard()


class TestProductRouting:
    def test_product_id_passed_to_generator(self):
        def echo_gen(product_id):
            val = float(product_id.split("_")[1])
            return {"x": torch.full((1,), val)}

        s = Supplier(echo_gen)
        pool = SupplierPool([s], prefetch=1)

        r1 = pool.get("model_1")
        pool.discard()
        r2 = pool.get("model_2")
        pool.discard()

        assert r1["x"].item() == 1.0
        assert r2["x"].item() == 2.0
