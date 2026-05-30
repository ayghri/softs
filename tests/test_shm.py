"""Tests for softs shared memory module."""
import pytest
import torch

from softs.market.mediums.shm import ShmMedium
from softs import BatchConfig, TensorSpec


class TestShmMedium:
    @pytest.fixture
    def config(self):
        return BatchConfig([
            TensorSpec("x", (3, 32, 32), "float32"),
            TensorSpec("y", (10,), "float32"),
        ])

    @pytest.fixture
    def owner(self, config):
        m = ShmMedium(address=None, slot_size=config.nbytes(), num_slots=4, create=True)
        yield m
        m.close()
        m.unlink()

    def test_create(self, owner, config):
        assert owner.num_slots == 4
        assert owner.slot_size == config.nbytes()

    def test_write_and_read(self, owner, config):
        writer = ShmMedium.attach(owner.address)
        try:
            data = config.encode(x=torch.randn(3, 32, 32), y=torch.randn(10))
            writer.write(0, data)
            tensors = config.decode(owner.read(0))
            assert tensors["x"].shape == (3, 32, 32)
            assert tensors["y"].shape == (10,)
        finally:
            writer.close()

    def test_attach(self, owner, config):
        other = ShmMedium.attach(owner.address)
        try:
            x, y = torch.randn(3, 32, 32), torch.randn(10)
            other.write(0, config.encode(x=x, y=y))
            tensors = config.decode(owner.read(0))
            torch.testing.assert_close(tensors["x"], x)
            torch.testing.assert_close(tensors["y"], y)
        finally:
            other.close()

    def test_context_manager(self, config):
        with ShmMedium(address=None, slot_size=config.nbytes(), num_slots=2, create=True) as m:
            assert m.num_slots == 2


class TestDtypeSupport:
    @pytest.fixture
    def create_pair(self):
        pairs = []
        def _create(config):
            owner = ShmMedium(address=None, slot_size=config.nbytes(), num_slots=2, create=True)
            writer = ShmMedium.attach(owner.address)
            pairs.append((owner, writer))
            return owner, writer
        yield _create
        for o, w in pairs:
            w.close(); o.close(); o.unlink()

    @pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
    def test_float_dtypes(self, create_pair, dtype):
        config = BatchConfig([TensorSpec("x", (10,), dtype), TensorSpec("y", (5,), dtype)])
        owner, writer = create_pair(config)
        x, y = torch.randn(10, dtype=getattr(torch, dtype)), torch.randn(5, dtype=getattr(torch, dtype))
        writer.write(0, config.encode(x=x, y=y))
        tensors = config.decode(owner.read(0))
        torch.testing.assert_close(tensors["x"], x)
        torch.testing.assert_close(tensors["y"], y)

    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
    def test_int_dtypes(self, create_pair, dtype):
        config = BatchConfig([TensorSpec("x", (10,), dtype), TensorSpec("y", (5,), dtype)])
        owner, writer = create_pair(config)
        x = torch.randint(-100, 100, (10,), dtype=getattr(torch, dtype))
        y = torch.randint(-100, 100, (5,), dtype=getattr(torch, dtype))
        writer.write(0, config.encode(x=x, y=y))
        tensors = config.decode(owner.read(0))
        torch.testing.assert_close(tensors["x"], x)
        torch.testing.assert_close(tensors["y"], y)

    def test_bfloat16(self, create_pair):
        config = BatchConfig([TensorSpec("x", (10,), "bfloat16"), TensorSpec("y", (5,), "bfloat16")])
        owner, writer = create_pair(config)
        x, y = torch.randn(10, dtype=torch.bfloat16), torch.randn(5, dtype=torch.bfloat16)
        writer.write(0, config.encode(x=x, y=y))
        tensors = config.decode(owner.read(0))
        torch.testing.assert_close(tensors["x"], x)
        torch.testing.assert_close(tensors["y"], y)
