"""Batch config, tensor specs, and encoding utilities."""

import math
from dataclasses import dataclass

import torch
import yaml

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
_TORCH_DTYPE_TO_STR: dict[torch.dtype, str] = {
    v: k for k, v in _DTYPE_MAP.items()
}
_DTYPE_SIZES: dict[torch.dtype, int] = {
    d: torch.empty(1, dtype=d).element_size() for d in _DTYPE_MAP.values()
}


def dtype_to_torch(dtype_str: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[dtype_str]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype_str}") from None


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    try:
        return _TORCH_DTYPE_TO_STR[dtype]
    except KeyError:
        raise ValueError(f"Unsupported torch dtype: {dtype}") from None


@dataclass
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def torch_dtype(self) -> torch.dtype:
        return dtype_to_torch(self.dtype)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * _DTYPE_SIZES[self.torch_dtype]

    def encode(self, tensor: torch.Tensor) -> bytes:
        if tuple(tensor.shape) != self.shape:
            raise ValueError(
                f"Shape mismatch: {tuple(tensor.shape)} != {self.shape}"
            )
        if tensor.dtype != self.torch_dtype:
            raise ValueError(
                f"Dtype mismatch: {tensor.dtype} != {self.torch_dtype}"
            )
        tensor = tensor.contiguous()
        if tensor.dtype == torch.bfloat16:
            return tensor.view(torch.uint16).numpy().tobytes()
        return tensor.numpy().tobytes()

    def decode(self, data: bytes) -> torch.Tensor:
        if len(data) != self.nbytes:
            raise ValueError(
                f"Data length {len(data)} != expected {self.nbytes}"
            )
        if self.dtype == "bfloat16":
            tensor = torch.frombuffer(bytearray(data), dtype=torch.uint16).view(
                torch.bfloat16
            )
        else:
            tensor = torch.frombuffer(bytearray(data), dtype=self.torch_dtype)
        return tensor.reshape(self.shape).clone()


@dataclass
class BatchConfig:
    specs: list[TensorSpec]

    def __post_init__(self):
        converted = []
        for spec in self.specs:
            if isinstance(spec, TensorSpec):
                converted.append(spec)
            elif hasattr(spec, "items"):
                converted.append(
                    TensorSpec(
                        **{
                            k: v
                            for k, v in spec.items()
                            if not str(k).startswith("_")
                        }
                    )
                )
            else:
                raise TypeError(
                    f"Expected TensorSpec or dict-like, got {type(spec)}"
                )
        object.__setattr__(self, "specs", converted)

        names = [s.name for s in self.specs]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate tensor names: {names}")

        self._name_to_idx = {s.name: i for i, s in enumerate(self.specs)}
        self._offsets = []
        off = 0
        for s in self.specs:
            self._offsets.append(off)
            off += s.nbytes

    @property
    def tensor_names(self) -> list[str]:
        return [s.name for s in self.specs]

    def nbytes(self) -> int:
        return sum(s.nbytes for s in self.specs)

    def get_spec(self, name: str) -> TensorSpec:
        idx = self._name_to_idx.get(name)
        if idx is None:
            raise KeyError(
                f"Unknown tensor '{name}'. Available: {self.tensor_names}"
            )
        return self.specs[idx]

    def get_offset(self, name: str) -> int:
        idx = self._name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"Unknown tensor '{name}'")
        return self._offsets[idx]

    def encode(self, **tensors: torch.Tensor) -> bytes:
        provided = set(tensors.keys())
        expected = set(self.tensor_names)
        if provided != expected:
            raise ValueError(
                f"Tensor mismatch: missing={expected - provided}, extra={provided - expected}"
            )
        return b"".join(spec.encode(tensors[spec.name]) for spec in self.specs)

    def decode(self, data: bytes) -> dict[str, torch.Tensor]:
        if len(data) != self.nbytes():
            raise ValueError(
                f"Data length {len(data)} != expected {self.nbytes()}"
            )
        return {
            spec.name: spec.decode(data[off : off + spec.nbytes])
            for spec, off in zip(self.specs, self._offsets)
        }

    def decode_single(self, data: bytes, name: str) -> torch.Tensor:
        spec = self.get_spec(name)
        off = self.get_offset(name)
        return spec.decode(data[off : off + spec.nbytes])

    @classmethod
    def from_dict(cls, config: dict) -> "BatchConfig":
        return cls(config.get("specs", []))

    @classmethod
    def from_yaml(cls, path: str) -> "BatchConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return {
            "specs": [
                {"name": s.name, "shape": list(s.shape), "dtype": s.dtype}
                for s in self.specs
            ]
        }


def make_xy_config(
    x_shape: tuple[int, ...],
    x_dtype: str,
    y_shape: tuple[int, ...],
    y_dtype: str,
) -> BatchConfig:
    return BatchConfig(
        [TensorSpec("x", x_shape, x_dtype), TensorSpec("y", y_shape, y_dtype)]
    )
