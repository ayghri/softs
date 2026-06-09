"""Shared pieces for the single-layer (qwen_layer0) distillation scripts.

The teacher captures a decoder layer's (input, output) hidden states with
:class:`~softs.catcher.ModelIOCatcher` and early-exits after that layer; it
serves any layer via one regex pattern. This example fixes the target to a
single layer (no switching) - the simplest end-to-end catcher distillation. See
``../layerwise`` for the all-layers version with per-layer ``set_model``.
"""

import logging

import torch
from omegaconf import DictConfig

from softs import BatchConfig, TensorSpec, EndpointConfig
from softs.catcher import ModelIOCatcher, parse_io_spec

logger = logging.getLogger(__name__)

# One regex serves any layer; the client orders a single concrete spec.
SUPPLIER_PATTERN = r"inputs\[model\.layers\.\d+\]\|outputs\[model\.layers\.\d+\]"


def layer_spec(idx: int) -> str:
    """The product_id a client orders to capture decoder layer ``idx``."""
    return f"inputs[model.layers.{idx}]|outputs[model.layers.{idx}]"


def build_endpoints() -> EndpointConfig:
    return EndpointConfig(
        frontend="ipc:///tmp/softs_qwen_fe.sock",
        backend="ipc:///tmp/softs_qwen_be.sock",
    )


def get_hidden_size(model_name: str) -> int:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_name).hidden_size


def build_config(cfg: DictConfig, hidden: int) -> BatchConfig:
    """One sample = (layer input, layer output) hidden states."""
    return BatchConfig(
        [
            TensorSpec("x", (cfg.model.seq_length, hidden), "bfloat16"),
            TensorSpec("y", (cfg.model.seq_length, hidden), "bfloat16"),
        ]
    )


def _hidden(obj):
    """Unwrap a decoder-layer in/out (often a 1-tuple) into the hidden tensor."""
    while isinstance(obj, (tuple, list)):
        obj = obj[0]
    return obj


def _input_hidden(entry: dict):
    """The hidden-state tensor a layer was called with (positional or kwarg)."""
    inp = entry["inputs"][0]
    if inp["args"]:
        return _hidden(inp["args"][0])
    return _hidden(inp["kwargs"]["hidden_states"])


class LayerIOTeacher:
    """Supplier ``generator_fn``: capture a decoder layer's (input, output).

    The ``product_id`` selects the layer; ``ModelIOCatcher`` captures it and
    early-exits, so only the prefix up to that layer runs.
    """

    def __init__(self, config, model_name, device, seq_len):
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = device
        self.seq_len = seq_len

        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
            .to(device)
            .eval()
        )
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.pad_token or tok.eos_token

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = " ".join(t for t in ds["text"] if t.strip())
        self._chunks = [
            tok(
                text[i : i + seq_len * 4],
                return_tensors="pt",
                max_length=seq_len,
                truncation=True,
                padding="max_length",
                return_token_type_ids=False,
            )
            for i in range(0, seq_len * 4 * 500, seq_len * 4)
        ]
        self._i = 0

        self.catcher = ModelIOCatcher(self.model, device)  # hooks attached in __init__

    @torch.no_grad()
    def __call__(self, product_id: str) -> bytes:
        toks = self._chunks[self._i % len(self._chunks)]
        self._i += 1
        toks = {k: v.to(self.device) for k, v in toks.items()}

        with self.catcher.for_product(product_id, early_exit=True) as buf:
            self.catcher.run(**toks, use_cache=False)

        entry = buf[parse_io_spec(product_id)[1][0]]
        x = _input_hidden(entry)[0]  # [seq, hidden]
        y = _hidden(entry["outputs"][0])[0]
        return self.config.encode(x=x.cpu(), y=y.cpu())
