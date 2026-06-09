"""Shared pieces for the distillation broker / supplier / client scripts.

``build_endpoints`` and ``build_config`` define the socket addresses and slot
layout that all three roles must agree on. ``TeacherGenerator`` is the Supplier's
``generator_fn``; the loss helpers are used by the student client.
"""

import logging

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from omegaconf import DictConfig

from softs import BatchConfig, TensorSpec, EndpointConfig

logger = logging.getLogger(__name__)


def build_endpoints() -> EndpointConfig:
    return EndpointConfig(
        frontend="ipc:///tmp/softs_distill_fe.sock",
        backend="ipc:///tmp/softs_distill_be.sock",
    )


def build_config(cfg: DictConfig) -> BatchConfig:
    """One sample = (input_ids, top_indices, soft_probs) with fixed shapes.

    Every role builds this identically from the same config so the slot layout
    (and ``slot_size``) matches across processes.
    """
    b, s, k = cfg.training.batch_size, cfg.model.seq_length, cfg.training.top_k
    return BatchConfig(
        [
            TensorSpec("input_ids", (b, s), "int64"),
            TensorSpec("top_indices", (b, s, k), "int64"),
            TensorSpec("soft_probs", (b, s, k), "float16"),
        ]
    )


def top_p_filter(logits: torch.Tensor, p: float, k: int, temperature: float):
    """Top-p filtered soft targets. Returns (indices, probs) both shape (B, S, k)."""
    scaled = logits.float() / temperature
    topk_logits, topk_indices = scaled.topk(k, dim=-1)
    topk_probs = F.softmax(topk_logits, dim=-1)
    sorted_probs, sorted_idx = topk_probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    mask = cumsum - sorted_probs <= p
    sorted_probs[~mask] = 0.0
    probs = sorted_probs.scatter(-1, sorted_idx, sorted_probs)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return topk_indices, probs


def soft_cross_entropy(student_logits, top_indices, soft_probs):
    """CE between student and teacher's top-p distribution."""
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    gathered = student_log_probs.gather(-1, top_indices)
    return -(soft_probs * gathered).sum(dim=-1).mean()


class TeacherGenerator:
    """Runs the teacher model and encodes top-p soft targets to bytes.

    Used as the Supplier's ``generator_fn``: called with a ``product_id`` and
    returns the encoded sample bytes for one slot.
    """

    def __init__(self, config, teacher_name, device, seq_len, batch_size,
                 top_k, top_p, temperature):
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        self.model = AutoModelForCausalLM.from_pretrained(
            teacher_name, dtype=torch.bfloat16
        ).to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", "sample-10BT",
            split="train", streaming=True,
        )
        self._ds_iter = iter(ds)
        self._token_buffer = torch.tensor([], dtype=torch.long)

    def _get_chunk(self) -> torch.Tensor:
        while len(self._token_buffer) < self.seq_len:
            try:
                sample = next(self._ds_iter)
            except StopIteration:
                self._ds_iter = iter(self._ds_iter)
                sample = next(self._ds_iter)
            ids = self.tokenizer(
                sample["text"], return_tensors="pt", truncation=False
            )["input_ids"][0]
            self._token_buffer = torch.cat([self._token_buffer, ids])
        chunk = self._token_buffer[: self.seq_len]
        self._token_buffer = self._token_buffer[self.seq_len :]
        return chunk

    def __call__(self, product_id: str) -> bytes:
        input_ids = torch.stack(
            [self._get_chunk() for _ in range(self.batch_size)]
        ).to(self.device)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with torch.no_grad():
                logits = self.model(input_ids, use_cache=False).logits

        indices, probs = top_p_filter(
            logits, p=self.top_p, k=self.top_k, temperature=self.temperature,
        )
        return self.config.encode(
            input_ids=input_ids.to(torch.int64).cpu(),
            top_indices=indices.to(torch.int64).cpu(),
            soft_probs=probs.to(torch.float16).cpu(),
        )
