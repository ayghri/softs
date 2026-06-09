#!/usr/bin/env python3
"""Distillation teacher supplier - run in its own process (typically one GPU).

    python supplier.py device.worker_gpu=0

Loads the teacher model and streams top-p soft targets through shared memory.
"""

import logging
import threading

import hydra
import torch
from omegaconf import DictConfig

from softs import Supplier, ShmMedium, setup_logging

from _common import TeacherGenerator, build_config, build_endpoints

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="distill")
def main(cfg: DictConfig):
    setup_logging(cfg.get("log_level", "INFO"))
    config = build_config(cfg)
    device = torch.device(
        f"cuda:{cfg.device.worker_gpu}" if torch.cuda.is_available() else "cpu"
    )
    teacher_name = cfg.model.teacher

    logger.info(f"Loading teacher: {teacher_name} on {device}")
    teacher = TeacherGenerator(
        config, teacher_name, device, cfg.model.seq_length,
        cfg.training.batch_size, cfg.training.top_k,
        cfg.training.top_p, cfg.training.temperature,
    )
    supplier = Supplier(
        generator_fn=teacher,
        product_ids=[teacher_name],
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
