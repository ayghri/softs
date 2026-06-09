"""Shared pieces for the layerwise distillation broker / supplier / client.

The teacher captures a decoder layer's (input, output) hidden states with
:class:`~softs.catcher.ModelIOCatcher`, selected per order by a ``product_id``
spec such as ``inputs[model.layers.0]|outputs[model.layers.0]``. With
``early_exit`` a request for layer N only runs the model *through* layer N. The
teacher registers a single regex (``SUPPLIER_PATTERN``) and so serves every layer
without enumerating them; the client orders one concrete spec per layer.
"""

import logging

import torch
from omegaconf import DictConfig

from softs import BatchConfig, TensorSpec, EndpointConfig
from softs.catcher import ModelIOCatcher, parse_io_spec

logger = logging.getLogger(__name__)

# One regex serves any layer; the client orders a concrete spec per layer.
SUPPLIER_PATTERN = r"inputs\[model\.layers\.\d+\]\|outputs\[model\.layers\.\d+\]"


def layer_spec(idx: int) -> str:
    """The product_id a client orders to capture decoder layer ``idx``."""
    return f"inputs[model.layers.{idx}]|outputs[model.layers.{idx}]"


def build_endpoints() -> EndpointConfig:
    return EndpointConfig(
        frontend="ipc:///tmp/softs_layerwise_fe.sock",
        backend="ipc:///tmp/softs_layerwise_be.sock",
    )


def get_model_info(model_name: str) -> dict:
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(model_name)
    return {"hidden_size": c.hidden_size, "num_layers": c.num_hidden_layers}


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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to(device)
        self.model.eval()

        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.pad_token or tok.eos_token

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = " ".join(t for t in ds["text"] if t.strip())
        self._chunks = []
        for i in range(0, len(text) - seq_len * 4, seq_len * 4):
            self._chunks.append(
                tok(
                    text[i : i + seq_len * 4],
                    return_tensors="pt",
                    max_length=seq_len,
                    truncation=True,
                    padding="max_length",
                    return_token_type_ids=False,
                )
            )
            if len(self._chunks) >= 2000:
                break
        self._i = 0

        self.catcher = ModelIOCatcher(self.model, device)

    @torch.no_grad()
    def __call__(self, product_id: str) -> bytes:
        tokens = self._chunks[self._i % len(self._chunks)]
        self._i += 1
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with self.catcher.for_product(product_id, early_exit=True) as buf:
            self.catcher.run(**tokens, use_cache=False)

        name = parse_io_spec(product_id)[1][0]  # the requested layer
        entry = buf[name]
        x = _input_hidden(entry)[0]  # [seq, hidden]
        y = _hidden(entry["outputs"][0])[0]
        return self.config.encode(x=x.cpu(), y=y.cpu())
