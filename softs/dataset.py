"""PyTorch Dataset and DataLoader for soft label generation."""

from typing import Callable, Iterator
import time
import multiprocessing
import ctypes

import torch
from torch.utils.data import DataLoader, IterableDataset

from .configs import BatchConfig
from .market import Client


class _SharedStr:
    """Process-safe mutable string via shared memory."""

    def __init__(self, value: str = "", max_len: int = 256):
        self._buf = multiprocessing.RawArray(ctypes.c_char, max_len)
        self._len = multiprocessing.RawValue(ctypes.c_int, 0)
        self._max = max_len
        if value:
            self.set(value)

    def set(self, value: str) -> None:
        encoded = value.encode()
        if len(encoded) > self._max:
            # Truncate on a valid UTF-8 boundary so get() never crashes.
            encoded = encoded[: self._max].decode("utf-8", "ignore").encode("utf-8")
        self._buf[: len(encoded)] = encoded
        self._len.value = len(encoded)

    def get(self) -> str:
        return bytes(self._buf[: self._len.value]).decode()


class Batch:
    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        slot_ids: list[int] | None = None,
        client: Client | None = None,
    ):
        self.tensors = tensors
        self.slot_ids = slot_ids or []
        self._client = client
        self._released = False

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def __contains__(self, key: str) -> bool:
        return key in self.tensors

    def keys(self):
        return self.tensors.keys()

    def release(self) -> None:
        if self._released or self._client is None:
            return
        for slot_id in self.slot_ids:
            self._client.release_slot(slot_id)
        self._released = True

    def __del__(self):
        self.release()


def make_collate_fn(
    client: Client, batch_config: BatchConfig, auto_release: bool = True
) -> Callable[[list[int]], Batch]:
    def collate_fn(slot_ids: list[int]) -> Batch:
        tensor_lists: dict[str, list[torch.Tensor]] = {
            name: [] for name in batch_config.tensor_names
        }
        for slot_id in slot_ids:
            for name, tensor in batch_config.decode(
                client.medium.read(slot_id)
            ).items():
                tensor_lists[name].append(tensor)
        batched = {name: torch.stack(ts) for name, ts in tensor_lists.items()}
        if auto_release:
            for slot_id in slot_ids:
                client.release_slot(slot_id)
            return Batch(batched, slot_ids, None)
        return Batch(batched, slot_ids, client)

    return collate_fn


class SoftIterableDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Infinite dataset yielding decoded tensor dicts.

    Call ``set_model(model_id)`` to switch models at any time.
    DataLoader workers detect the change and discard pending work.
    """

    def __init__(
        self,
        model_id: str,
        endpoint: str,
        batch_config: BatchConfig,
        medium_cls,
        num_slots: int = 8,
        retry_delay: float = 0.01,
        request_timeout: float | None = 60.0,
    ):
        self._model = _SharedStr(model_id)
        self.endpoint = endpoint
        self.num_slots = num_slots
        self.batch_config = batch_config
        self.medium_cls = medium_cls
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout
        self._client: Client | None = None

    @property
    def model_id(self) -> str:
        return self._model.get()

    def set_model(self, model_id: str) -> None:
        self._model.set(model_id)

    def _ensure_client(self) -> Client:
        if self._client is None:
            self._client = Client(
                endpoint=self.endpoint,
                slot_size=self.batch_config.nbytes(),
                medium_cls=self.medium_cls,
                num_slots=self.num_slots,
            )
            self._client.hello()
        return self._client

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        client = self._ensure_client()
        current = self.model_id
        waited = 0.0
        while True:
            wanted = self.model_id
            if wanted != current:
                client.discard()
                current = wanted
                waited = 0.0
            slot_id = client.request_sample(
                current, timeout_ms=int(self.retry_delay * 1000)
            )
            if slot_id is None:
                # Fail loudly instead of hanging forever on a dead broker/supplier.
                waited += self.retry_delay
                if self.request_timeout is not None and waited >= self.request_timeout:
                    raise RuntimeError(
                        f"No sample for product '{current}' within "
                        f"{self.request_timeout}s - is the broker running and a "
                        f"supplier serving '{current}'?"
                    )
                time.sleep(self.retry_delay)
                continue
            waited = 0.0
            tensors = self.batch_config.decode(client.medium.read(slot_id))
            client.release_slot(slot_id)
            yield tensors

    def __del__(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass


class SoftDataLoader(DataLoader):
    """DataLoader with model switching support.

    Usage::

        loader = SoftDataLoader(model_id="teacher_v1", endpoint=ep.frontend,
                                batch_config=config, medium_cls=ShmMedium,
                                num_slots=8, batch_size=4)
        for batch in loader:
            train(batch)

        loader.set_model("teacher_v2")  # switch the requested product id
    """

    def __init__(
        self,
        model_id: str,
        endpoint: str,
        batch_config: BatchConfig,
        medium_cls,
        num_slots: int = 8,
        retry_delay: float = 0.01,
        request_timeout: float | None = 60.0,
        **dataloader_kwargs,
    ):
        self.soft_dataset = SoftIterableDataset(
            model_id=model_id,
            endpoint=endpoint,
            batch_config=batch_config,
            medium_cls=medium_cls,
            num_slots=num_slots,
            retry_delay=retry_delay,
            request_timeout=request_timeout,
        )
        super().__init__(self.soft_dataset, **dataloader_kwargs)

    def set_model(self, model_id: str) -> None:
        self.soft_dataset.set_model(model_id)

    @property
    def model_id(self) -> str:
        return self.soft_dataset.model_id
