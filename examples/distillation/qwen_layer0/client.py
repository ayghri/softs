#!/usr/bin/env python3
"""Single-layer distillation student client - run in its own process (one GPU).

    python client.py device.student_gpu=0

Trains a copy of one decoder layer to map the teacher's captured (input ->
output) hidden states, pulled with ``SoftDataLoader`` (fixed product id, so no
layer switching - the simplest end-to-end catcher example).
"""

import logging

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from tqdm import tqdm

from softs import ShmMedium, SoftDataLoader, setup_logging

from _common import build_config, build_endpoints, get_hidden_size, layer_spec

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    setup_logging(cfg.get("log_level", "INFO"))
    model_name = cfg.model.name
    seq_len = cfg.model.seq_length
    layer_idx = cfg.model.layer
    config = build_config(cfg, get_hidden_size(model_name))
    device = torch.device(
        f"cuda:{cfg.device.student_gpu}" if torch.cuda.is_available() else "cpu"
    )

    from transformers import AutoModelForCausalLM
    logger.info(f"Loading student: {model_name} on {device}")
    student = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    ).to(device)
    rotary_emb = student.model.rotary_emb.to(device)
    criterion = nn.MSELoss()

    layer = student.model.layers[layer_idx].to(device)
    for p in layer.parameters():
        p.requires_grad = False
    for m in layer.modules():
        if isinstance(m, nn.Linear):
            for p in m.parameters():
                p.requires_grad = True
    optimizer = AdamW(
        [p for p in layer.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
    )
    layer.train()

    loader = SoftDataLoader(
        model_id=layer_spec(layer_idx),
        endpoint=build_endpoints().frontend,
        batch_config=config,
        medium_cls=ShmMedium,
        num_slots=cfg.softs.get("slot_count", 8),
        batch_size=cfg.training.batch_size,
    )

    logger.info(f"Distilling layer {layer_idx} for {cfg.training.steps} steps")
    total_loss, count = 0.0, 0
    pbar = tqdm(range(cfg.training.steps), desc=f"Layer {layer_idx}")
    for _, batch in zip(pbar, loader):
        x = batch["x"].to(device)  # (B, seq, hidden)
        y = batch["y"].to(device)

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = rotary_emb(x, position_ids)
        out = layer(x, position_embeddings=position_embeddings)
        pred = out[0] if isinstance(out, tuple) else out

        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1
        pbar.set_postfix(loss=f"{total_loss / max(count, 1):.4e}")

    logger.info(f"Done: layer {layer_idx} loss={total_loss / max(count, 1):.4e}")


if __name__ == "__main__":
    main()
