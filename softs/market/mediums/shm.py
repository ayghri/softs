"""Shared memory transfer medium."""

import logging
import secrets
from multiprocessing import shared_memory

from .base import Medium

logger = logging.getLogger(__name__)


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

    def write(self, slot_id: int, data: bytes) -> bool:

        try:
            buf = self._shm.buf
            buf[slot_id : slot_id + len(data)] = data
            return True
        except Exception:
            logger.warning(
                "shm write failed at offset %d (len %d) on '%s'",
                slot_id,
                len(data),
                self.address,
                exc_info=True,
            )
            return False

    def read(self, slot_id: int) -> bytes:
        off = slot_id * self.slot_size
        return bytes(self._shm.buf[off : off + self.slot_size])

    @classmethod
    def attach(cls, address: str) -> "ShmMedium":
        return cls(address=address, slot_size=0, num_slots=0, create=False)

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        if self.is_owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
