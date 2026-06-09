#!/usr/bin/env python3
"""Layerwise distillation teacher supplier - run in its own process (one GPU).

    python supplier.py device.worker_gpu=0

Captures decoder-layer (input, output) hidden states with ModelIOCatcher and
serves any layer via a single regex pattern.
"""

import logging
import threading

import hydra
import torch
from omegaconf import DictConfig

from softs import Supplier, ShmMedium, setup_logging

from _common import (
    LayerIOTeacher,
    SUPPLIER_PATTERN,
    build_config,
    build_endpoints,
    get_model_info,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    setup_logging(cfg.get("log_level", "INFO"))
    model_name = cfg.model.name
    info = get_model_info(model_name)
    config = build_config(cfg, info["hidden_size"])
    device = torch.device(
        f"cuda:{cfg.device.worker_gpu}" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Loading teacher: {model_name} on {device}")
    teacher = LayerIOTeacher(config, model_name, device, cfg.model.seq_length)
    supplier = Supplier(
        generator_fn=teacher,
        product_ids=[SUPPLIER_PATTERN],  # regex: serves every layer
        endpoint=build_endpoints().backend,
        medium_cls=ShmMedium,
        slot_size=config.nbytes(),
    )
    supplier.start()
    logger.info("Teacher supplier running. Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        supplier.stop()


if __name__ == "__main__":
    main()
