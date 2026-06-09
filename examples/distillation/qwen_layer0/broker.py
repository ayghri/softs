#!/usr/bin/env python3
"""Single-layer distillation broker - run in its own process/terminal.

    python broker.py
"""

import logging
import threading

import hydra
from omegaconf import DictConfig

from softs import Broker, setup_logging

from _common import build_endpoints

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    setup_logging(cfg.get("log_level", "INFO"))
    broker = Broker(endpoints=build_endpoints())
    broker.start()
    logger.info("Broker running. Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        broker.stop()


if __name__ == "__main__":
    main()
