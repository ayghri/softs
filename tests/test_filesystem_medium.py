"""Tests for FilesystemMedium."""
import tempfile
import time
from pathlib import Path

import pytest
import torch

from softs import (
    Broker, Supplier, Client,
    BatchConfig, TensorSpec, FilesystemMedium, EndpointConfig,
)

SLOT_SIZE = 1024
BUFFER_N = 4

ENDPOINTS = EndpointConfig(
    frontend="ipc:///tmp/sl_test_fs_fe.sock",
    backend="ipc:///tmp/sl_test_fs_be.sock",
)


@pytest.fixture
def tmp_path_factory():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestFilesystemMedium:
    def test_create_and_read_write(self, tmp_path_factory):
        path = tmp_path_factory / "test.bin"
        reader = FilesystemMedium(address=str(path), slot_size=SLOT_SIZE, num_slots=BUFFER_N, create=True)
        writer = FilesystemMedium(address=str(path), slot_size=SLOT_SIZE, num_slots=BUFFER_N, create=False)
        data = b"\xab" * SLOT_SIZE
        writer.write(0, data)
        assert reader.read(0) == data
        writer.close()
        reader.close()
        reader.unlink()

    def test_attach(self, tmp_path_factory):
        path = tmp_path_factory / "attach.bin"
        reader = FilesystemMedium(address=str(path), slot_size=SLOT_SIZE, num_slots=BUFFER_N, create=True)
        writer = FilesystemMedium.attach(str(path))
        assert writer.write(0, b"\xcd" * 100)
        writer.close()
        reader.close()
        reader.unlink()

    def test_multiple_slots(self, tmp_path_factory):
        path = tmp_path_factory / "multi.bin"
        reader = FilesystemMedium(address=str(path), slot_size=SLOT_SIZE, num_slots=BUFFER_N, create=True)
        writer = FilesystemMedium(address=str(path), slot_size=SLOT_SIZE, num_slots=BUFFER_N, create=False)
        for i in range(BUFFER_N):
            writer.write(i * SLOT_SIZE, bytes([i]) * SLOT_SIZE)
        for i in range(BUFFER_N):
            assert reader.read(i) == bytes([i]) * SLOT_SIZE
        writer.close()
        reader.close()
        reader.unlink()

    def test_context_manager(self, tmp_path_factory):
        path = tmp_path_factory / "ctx.bin"
        m = FilesystemMedium(address=str(path), slot_size=SLOT_SIZE, num_slots=BUFFER_N, create=True)
        assert Path(path).exists()
        m.close()
        m.unlink()
        assert not Path(path).exists()


class TestFilesystemSupplierIntegration:
    def test_full_pipeline(self, tmp_path_factory):
        config = BatchConfig([TensorSpec("x", (3, 8, 8), "float32"), TensorSpec("y", (4,), "float32")])
        b = Broker(endpoints=ENDPOINTS); b.start(); time.sleep(0.1)
        try:
            s = Supplier(
                medium_cls=FilesystemMedium,
                generator_fn=lambda mid: config.encode(x=torch.randn(3, 8, 8), y=torch.randn(4)),
                product_ids=["fs_test"],
                slot_size=config.nbytes(),
                endpoint=ENDPOINTS.backend,
            )
            s.start(); time.sleep(0.1); s.stop()
        finally:
            b.stop()
