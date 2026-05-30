"""PyTorch Dataset and DataLoader for soft label generation."""

import ctypes
import multiprocessing
import time
from typing import Callable, Iterator

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
        encoded = value.encode()[: self._max]
        self._buf[: len(encoded)] = encoded
        self._len.value = len(encoded)

    def get(self) -> str:
        return bytes(self._buf[: self._len.value]).decode()


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
        max_retries: int = 10,
        retry_delay: float = 0.01,
    ):
        self._model = _SharedStr(model_id)
        self.endpoint = endpoint
        self.num_slots = num_slots
        self.batch_config = batch_config
        self.medium_cls = medium_cls
        self.max_retries = max_retries
        self.retry_delay = retry_delay
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
        retries = 0
        current = self.model_id
        while True:
            wanted = self.model_id
            if wanted != current:
                client.discard()
                current = wanted
                retries = 0
            slot_id = client.request_sample(
                current, timeout_ms=int(self.retry_delay * 1000)
            )
            if slot_id is None:
                retries += 1
                if retries >= self.max_retries:
                    time.sleep(self.retry_delay)
                continue
            retries = 0
            tensors = self.batch_config.decode(client.medium.read(slot_id))
            client.release_slot(slot_id)
            yield tensors

    def __del__(self):
        if self._client is not None:
            self._client.close()


class SoftDataLoader(DataLoader):
    """DataLoader with model switching support.

    Usage::

        loader = SoftDataLoader(model_id="teacher_v1", slot_count=8,
                                batch_config=config, endpoints=ep,
                                medium_cls=ShmMedium, batch_size=4)
        for batch in loader:
            train(batch)

        loader.set_model("teacher_v2")  # all workers switch automatically
    """

    def __init__(
        self,
        model_id: str,
        endpoint: str,
        batch_config: BatchConfig,
        medium_cls,
        num_slots: int = 8,
        max_retries: int = 10,
        retry_delay: float = 0.01,
        **dataloader_kwargs,
    ):
        self.dataset = SoftIterableDataset(
            model_id=model_id,
            endpoint=endpoint,
            batch_config=batch_config,
            medium_cls=medium_cls,
            num_slots=num_slots,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        super().__init__(self.dataset, **dataloader_kwargs)

    def set_model(self, model_id: str) -> None:
        self.dataset.set_model(model_id)

    @property
    def model_id(self) -> str:
        return self.dataset.model_id


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
