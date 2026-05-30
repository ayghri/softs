"""Base class for transfer mediums."""

from abc import ABC, abstractmethod


class Medium(ABC):
    """A fixed-size slot buffer that suppliers write to and clients read from.

    The broker routes opaque (buf_name, slot_offset) between them.
    Subclasses implement the actual data transfer mechanism.
    """

    def __init__(
        self,
        address: str,
        slot_size: int,
        num_slots: int,
        create=False,
        **kwargs,
    ):
        self.address = address
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.is_owner = create

    @classmethod
    @abstractmethod
    def attach(cls, address: str) -> "Medium": ...

    @abstractmethod
    def write(self, slot_index: int, data: bytes) -> bool:
        """Write data at byte offset. Returns False if resource is gone."""
        ...

    @abstractmethod
    def read(self, slot_index: int) -> bytes: ...

    def close(self) -> None:
        pass

    def unlink(self) -> None:
        """Remove underlying resource (shm segment, file, etc.)."""

    def __enter__(self) -> "Medium":
        return self

    def __exit__(self, *args) -> None:
        self.close()
