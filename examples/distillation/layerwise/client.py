#!/usr/bin/env python3
"""Layerwise distillation student client - run in its own process (one GPU).

    python client.py device.student_gpu=0

Trains the student layer by layer. It uses a single ``SoftDataLoader`` and calls
``set_model`` to point at the next layer's spec; the loader discards in-flight
work from the previous layer (generation fencing), so stale activations are never
trained on.
"""

import logging

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from tqdm import tqdm

from softs import ShmMedium, SoftDataLoader, setup_logging

from _common import build_config, build_endpoints, get_model_info, layer_spec

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    setup_logging(cfg.get("log_level", "INFO"))
    model_name = cfg.model.name
    seq_len = cfg.model.seq_length
    info = get_model_info(model_name)
    num_layers = info["num_layers"]
    config = build_config(cfg, info["hidden_size"])
    device = torch.device(
        f"cuda:{cfg.device.student_gpu}" if torch.cuda.is_available() else "cpu"
    )
    batch_size = cfg.training.batch_size

    from transformers import AutoModelForCausalLM
    logger.info(f"Loading student: {model_name} on {device}")
    student = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    ).to(device)
    rotary_emb = student.model.rotary_emb.to(device)
    criterion = nn.MSELoss()

    # One loader, switched per layer via set_model (discards stale work on switch).
    loader = SoftDataLoader(
        model_id=layer_spec(0),
        endpoint=build_endpoints().frontend,
        batch_config=config,
        medium_cls=ShmMedium,
        num_slots=cfg.softs.get("slot_count", 16),
        batch_size=batch_size,
    )
    batches = iter(loader)

    steps_per_layer = cfg.training.samples_per_layer * cfg.training.epochs_per_layer

    for layer_idx in range(num_layers):
        logger.info(f"Training layer {layer_idx}")
        loader.set_model(layer_spec(layer_idx))  # next batch fences out old work

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

        total_loss, count = 0.0, 0
        pbar = tqdm(range(steps_per_layer), desc=f"Layer {layer_idx}")
        for _ in pbar:
            batch = next(batches)
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

        logger.info(
            f"Layer {layer_idx}: loss={total_loss / max(count, 1):.4e}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
