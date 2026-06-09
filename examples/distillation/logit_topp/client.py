#!/usr/bin/env python3
"""Distillation student client - run in its own process (typically one GPU).

    python client.py device.student_gpu=1 training.steps=500

Pulls soft-target batches with ``SoftDataLoader`` (the package's PyTorch
integration; each slot already holds a full ``(batch, seq, ...)`` tensor, so the
loader uses ``batch_size=None``) and trains the student with soft cross-entropy.
"""

import logging

import hydra
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from omegaconf import DictConfig
from torch.optim import AdamW
from tqdm import tqdm

from softs import ShmMedium, SoftDataLoader, setup_logging

from _common import build_config, build_endpoints, soft_cross_entropy

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="distill")
def main(cfg: DictConfig):
    setup_logging(cfg.get("log_level", "INFO"))
    config = build_config(cfg)
    device = torch.device(
        f"cuda:{cfg.device.student_gpu}" if torch.cuda.is_available() else "cpu"
    )
    grad_accum = cfg.training.grad_accum

    from transformers import AutoModelForCausalLM
    logger.info(f"Loading student: {cfg.model.student} on {device}")
    student = AutoModelForCausalLM.from_pretrained(
        cfg.model.student, dtype=torch.bfloat16
    ).to(device)
    student.train()
    optimizer = AdamW(student.parameters(), lr=cfg.training.learning_rate)

    # Each slot already holds a full batch, so batch_size=None yields it as-is.
    loader = SoftDataLoader(
        model_id=cfg.model.teacher,
        endpoint=build_endpoints().frontend,
        batch_config=config,
        medium_cls=ShmMedium,
        num_slots=cfg.softs.get("slot_count", 4),
        batch_size=None,
        request_timeout=300.0,
    )

    total_tokens = cfg.training.steps * cfg.training.batch_size * cfg.model.seq_length
    logger.info(
        f"Training: {cfg.training.steps} steps, {total_tokens / 1e6:.1f}M tokens, "
        f"batch={cfg.training.batch_size}x{grad_accum}accum, "
        f"seq_len={cfg.model.seq_length}, top_k={cfg.training.top_k}"
    )

    pbar = tqdm(range(cfg.training.steps), desc="Distilling")
    running_loss = 0.0
    optimizer.zero_grad()

    for step, data in zip(pbar, loader):
        input_ids = data["input_ids"].to(device)
        top_indices = data["top_indices"].to(device)
        soft_probs = data["soft_probs"].float().to(device)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            student_logits = student(input_ids, use_cache=False).logits
        loss = soft_cross_entropy(student_logits, top_indices, soft_probs) / grad_accum
        loss.backward()

        running_loss += loss.item() * grad_accum

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % cfg.training.log_every == 0:
            avg = running_loss / cfg.training.log_every
            pbar.set_postfix(loss=f"{avg:.4f}")
            running_loss = 0.0

    logger.info("Training complete!")
    if cfg.training.save_path:
        student.save_pretrained(cfg.training.save_path)
        logger.info(f"Saved to {cfg.training.save_path}")


if __name__ == "__main__":
    main()
