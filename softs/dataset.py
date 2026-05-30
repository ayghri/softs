"""PyTorch Dataset and DataLoader for async soft label generation."""

import ctypes
import multiprocessing
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from .supplier import SupplierPool


class _SharedStr:
    """Process-safe mutable string via shared memory."""

    def __init__(self, value: str = "", max_len: int = 256):
        self._buf = multiprocessing.RawArray(ctypes.c_char, max_len)
        self._len = multiprocessing.RawValue(ctypes.c_int, 0)
        self._max = max_len
        if value:
            self.set(value)

    def set(self, value: str) -> None:
        encoded = value.encode()[: self._max]
        self._buf[: len(encoded)] = encoded
        self._len.value = len(encoded)

    def get(self) -> str:
        return bytes(self._buf[: self._len.value]).decode()


class SoftIterableDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Infinite dataset that pulls samples from a SupplierPool.

    Args:
        pool: SupplierPool that manages generation and prefetching.
        model_id: Initial model/product ID to request.
        timeout: Seconds to wait per sample (None = block forever).
    """

    def __init__(self, pool: SupplierPool, model_id: str, timeout: float | None = None):
        self._pool = pool
        self._model = _SharedStr(model_id)
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model.get()

    def set_model(self, model_id: str) -> None:
        """Switch all suppliers to a new model.

        Tells every supplier actor to load the new model,
        drops stale prefetched results, and updates model_id.
        """
        self._pool.set_model(model_id)
        self._model.set(model_id)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        current = self.model_id
        while True:
            wanted = self.model_id
            if wanted != current:
                self._pool.discard()
                current = wanted
            sample = self._pool.get(current, timeout=self._timeout)
            if sample is not None:
                yield sample


class SoftDataLoader(DataLoader):
    """DataLoader that consumes from a SupplierPool with model switching.

    Usage::

        suppliers = [Supplier(TeacherGen, num_gpus=1) for _ in range(2)]
        pool = SupplierPool(suppliers, prefetch=4)
        loader = SoftDataLoader(pool, model_id="teacher_v1", batch_size=4)

        for batch in loader:
            train(batch)

        loader.set_model("teacher_v2")  # all workers switch
    """

    def __init__(
        self,
        pool: SupplierPool,
        model_id: str,
        timeout: float | None = None,
        **dataloader_kwargs,
    ):
        self._soft_dataset = SoftIterableDataset(pool, model_id, timeout)
        super().__init__(self._soft_dataset, **dataloader_kwargs)

    def set_model(self, model_id: str) -> None:
        self._soft_dataset.set_model(model_id)

    @property
    def model_id(self) -> str:
        return self._soft_dataset.model_id
