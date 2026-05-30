"""Shared memory transfer medium."""

import secrets
from multiprocessing import shared_memory

from .base import Medium


class ShmMedium(Medium):

    def __init__(
        self,
        address,
        slot_size,
        num_slots,
        create=False,
        **kwargs,
    ):
        if address is None:
            if create:
                address = f"sl_{secrets.token_hex(8)}"
            else:
                raise ValueError("address required when create=False")
        super().__init__(address, slot_size, num_slots, create)
        self._shm = shared_memory.SharedMemory(
            name=address, create=create, size=slot_size * num_slots
        )

    def write(self, slot_offset: int, data: bytes) -> bool:
        try:
            buf = self._shm.buf
            buf[slot_offset : slot_offset + len(data)] = data  # type: ignore
            return True
        except Exception:
            return False

    def read(self, slot_id: int) -> bytes:
        off = slot_id * self.slot_size
        return bytes(self._shm.buf[off : off + self.slot_size])  # type: ignore

    @classmethod
    def attach(cls, address: str) -> "ShmMedium":
        return cls(address=address, slot_size=0, num_slots=0, create=False)

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        if self.is_owner:
            self._shm.unlink()
