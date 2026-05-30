"""Soft labels: async on-the-fly training data generation for PyTorch."""

import logging

__version__ = "0.6.0"


def setup_logging(level: int | str = logging.INFO) -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


from .configs import BatchConfig, TensorSpec, make_xy_config
from .supplier import Supplier, SupplierPool
from .dataset import SoftIterableDataset, SoftDataLoader
