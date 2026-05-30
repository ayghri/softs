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


# User-facing
from .configs import BatchConfig, TensorSpec, make_xy_config
from .market import EndpointConfig
from .dataset import SoftIterableDataset, SoftDataLoader, Batch, make_collate_fn

# Market internals (for advanced use)
from .market import Broker, Supplier, Client, BrokerStats
from .market import ShmMedium, FilesystemMedium, TCPMedium
