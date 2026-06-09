"""File-backed transfer medium using memory-mapped files."""

import logging
import mmap
import os
from pathlib import Path

from .base import Medium

logger = logging.getLogger(__name__)


class FilesystemMedium(Medium):
    """buf_name is the file path."""

    def __init__(
        self,
        address,
        slot_size,
        num_slots,
        create=False,
        **kwargs,
    ):
        super().__init__(address, slot_size, num_slots, create)

        if isinstance(address, str):
            self.path = Path(address)
        else:
            raise ValueError(f"address must be str or Path, got {address}")

        total = slot_size * num_slots

        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as f:
                f.truncate(total)

        self._fd = os.open(str(self.path), os.O_RDWR)
        size = total if create else os.fstat(self._fd).st_size
        self._mmap = mmap.mmap(self._fd, size, access=mmap.ACCESS_WRITE)

    def write(self, slot_index: int, data: bytes) -> bool:
        try:
            self._mmap[slot_index : slot_index + len(data)] = data  # type: ignore
            return True
        except Exception:
            logger.warning(
                "filesystem write failed at offset %d (len %d) on '%s'",
                slot_index,
                len(data),
                self.address,
                exc_info=True,
            )
            return False

    def read(self, slot_index: int) -> bytes:
        off = slot_index * self.slot_size
        return bytes(self._mmap[off : off + self.slot_size])  # type: ignore

    def close(self) -> None:
        try:
            if self._mmap:
                self._mmap.close()
        finally:
            self._mmap = None
        try:
            if self._fd >= 0:
                os.close(self._fd)
        finally:
            self._fd = -1

    @classmethod
    def attach(cls, address: str) -> "FilesystemMedium":
        return cls(address=address, slot_size=0, num_slots=0, create=False)

    def unlink(self) -> None:
        if self.is_owner and self.path.exists():
            self.path.unlink()
