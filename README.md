# softs

A broker-based data pipeline for distributed teacher-student training in PyTorch.

## Overview

`softs` provides a **data-agnostic** message routing system for teacher-student workflows:

- **Broker**: Routes messages between students and workers. Knows nothing about the data.
- **Workers**: Generate samples on-demand, write raw bytes to student-owned memory.
- **Students**: Own memory slots, request samples, read and decode bytes.

The library only moves bytes. What those bytes represent is entirely up to your application. Use `BatchConfig` for PyTorch tensor encoding/decoding.

## Key Features

- **Zero-copy transfer**: Workers write directly to student shared memory
- **Async pipeline**: Students train while workers generate the next batch
- **Model switching**: Change teacher models mid-training (e.g., layer-by-layer distillation)
- **Fault tolerance**: Workers/students can crash and restart independently
- **DDP compatible**: Works with PyTorch's DistributedDataParallel

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           BROKER                                 │
│                   (message router, data-agnostic)                │
│                                                                  │
│   Frontend        Backend         Control        ControlPub      │
│   (ROUTER)        (ROUTER)       (ROUTER)         (PUB)          │
│      ▲               ▲              ▲               │            │
└──────┼───────────────┼──────────────┼───────────────┼────────────┘
       │               │              │               │
  ┌────┴────┐    ┌─────┴─────┐   ┌────┴────┐    ┌────┴────┐
  │ Student │    │  Worker   │   │Student 0│    │ Workers │
  │DataLoader    │ (teacher) │   │(leader) │    │  (SUB)  │
  │ workers │    │           │   │         │    │         │
  └─────────┘    └───────────┘   └─────────┘    └─────────┘
```

### Message Flow

1. **Student** creates shared memory slots and sends `REQUEST` with `{token, shm_name, slot_offset}`
2. **Broker** queues the request, assigns it to an available **Worker** via `WORK`
3. **Worker** generates sample bytes using your `generator_fn`, writes directly to shared memory
4. **Worker** sends `DONE` to broker, broker sends `COMPLETE` to student
5. **Student** reads bytes from shared memory, decodes tensors, trains

### Model Switching

Student rank 0 (leader) can change the model at any time:

```python
client.set_model("layer_5")  # Workers now generate for layer_5
```

This:
1. Increments a **generation counter**
2. Broadcasts new model to all workers via PUB/SUB
3. Discards any pending work from the old generation
4. Workers start generating for the new model immediately

## Installation

```bash
pip install softs
# or
poetry add softs
```

**Dependencies**: `pyzmq`, `torch`, `numpy`

## Quick Start

### 1. Define your data format with BatchConfig

```python
from softs import BatchConfig, TensorSpec

config = BatchConfig([
    TensorSpec("x", (3, 224, 224), "float32"),
    TensorSpec("y", (1000,), "float32"),
])
```

### 2. Start the broker

```python
from softs import Broker, setup_logging

setup_logging("INFO")
Broker().run()
```

### 3. Start worker(s)

```python
import torch
from softs import Worker, setup_logging

setup_logging("INFO")

def generate_sample(model_id: str, model_cfg: dict | None) -> bytes:
    # Your generation logic - runs once per sample
    x = torch.randn(3, 224, 224)
    y = torch.randn(1000)
    return config.encode(x=x, y=y)

Worker(
    generator_fn=generate_sample,
    slot_size=config.nbytes(),
).run()
```

### 4. Run student training

```python
from softs import StudentClient, DistillIterableDataset, setup_logging

setup_logging("INFO")

client = StudentClient(
    student_rank=0,
    slot_count=16,
    batch_config=config,
)
client.hello()
client.set_model("my_model")

dataset = DistillIterableDataset(
    student_rank=0,
    generation_value=client.generation_value,
    slot_count=8,
    batch_config=config,
)

for batch in dataset:
    x, y = batch["x"], batch["y"]
    # Training loop...

client.close()
```

## BatchConfig API

`BatchConfig` describes tensors and handles encoding/decoding:

```python
from softs import BatchConfig, TensorSpec

# Define specs
config = BatchConfig([
    TensorSpec("hidden", (512, 768), "bfloat16"),
    TensorSpec("labels", (512,), "int64"),
])

# Total bytes
config.nbytes()  # -> 790528

# Encode tensors to bytes
data = config.encode(hidden=hidden_tensor, labels=label_tensor)

# Decode bytes to dict of tensors
tensors = config.decode(data)

# Decode a single tensor
hidden = config.decode_single(data, "hidden")

# Properties
config.tensor_names  # ['hidden', 'labels']
config.get_spec("hidden")  # TensorSpec object
```

**Supported dtypes**: `float64`, `float32`, `float16`, `bfloat16`, `int64`, `int32`, `int16`, `int8`, `uint8`, `bool`

### Loading from YAML

```python
config = BatchConfig.from_yaml("config.yaml")

# Or from dict
config = BatchConfig.from_dict({
    "specs": [
        {"name": "x", "shape": [512, 768], "dtype": "bfloat16"},
        {"name": "y", "shape": [512, 768], "dtype": "bfloat16"},
    ]
})
```

### Hydra Integration

```yaml
# config.yaml
batch_config:
  _target_: softs.BatchConfig
  specs:
    - name: x
      shape: [512, 768]
      dtype: bfloat16
```

```python
from hydra.utils import instantiate
config = instantiate(cfg.batch_config)
```

## Transfer Mediums

By default, softs uses **POSIX shared memory** for zero-copy data transfer. The architecture supports other mediums through the `Medium` protocol.

### How Mediums Work

1. **Students** create and own the medium (e.g., shared memory segment)
2. **Broker** routes opaque addressing info (`shm_name`, `slot_offset`) to workers
3. **Workers** write directly to the medium using the addressing info
4. **Students** read from the medium after receiving completion notification

The broker never touches the actual data - it only routes metadata.

### Default: SharedMemoryManager

```python
from softs.mediums import SharedMemoryManager

# Students create shm (read_only=True = create owner)
shm = SharedMemoryManager(slot_count=16, slot_stride=1024, read_only=True)

# Workers attach by name (read_only=False = attach)
shm = SharedMemoryManager(slot_count=16, slot_stride=1024, read_only=False, run_id=run_id)
```

### Custom Mediums

Implement the `Medium` protocol or extend `MediumBase`:

```python
from softs.mediums import MediumBase

class FileMedium(MediumBase):
    """File-based medium (example for network filesystems)."""

    def __init__(self, slot_count: int, slot_stride: int, path: str):
        self._path = path
        self._slot_count = slot_count
        self._slot_stride = slot_stride
        self._file = open(path, 'w+b')
        self._file.truncate(slot_count * slot_stride)

    @property
    def buf_name(self) -> str:
        return self._path

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def slot_stride(self) -> int:
        return self._slot_stride

    def read_slot_tensors(self, slot_id: int) -> bytes:
        offset = slot_id * self._slot_stride
        self._file.seek(offset)
        return self._file.read(self._slot_stride)

    def close(self) -> None:
        self._file.close()

    def unlink(self) -> None:
        import os
        os.unlink(self._path)
```

Potential medium implementations:
- **GPU Direct**: Use CUDA IPC for GPU-to-GPU transfer
- **Network**: Use RDMA or TCP for multi-node setups
- **Memory-mapped files**: For persistence or network filesystems

## Running with DDP

```bash
# Terminal 1: Broker
python -c "from softs import Broker; Broker().run()"

# Terminal 2: Worker(s) - can run multiple
python worker.py

# Terminal 3: DDP students
torchrun --nproc_per_node=2 student.py
```

For multi-GPU students:
- Only rank 0 creates the main `StudentClient` and calls `set_model()`
- Other ranks listen for model changes via PUB/SUB

```python
if rank == 0:
    client.set_model("layer_0")
else:
    client.start_sub_listener()
```

## Example: Layer-by-Layer LLM Distillation

See `examples/distill_llm.py` for a complete example that:
1. Loads a teacher LLM
2. Distills layer-by-layer (switches model per layer)
3. Uses Hydra for configuration
4. Supports DDP training

```bash
# Start broker
python distill_llm.py mode=broker

# Start worker (loads teacher model)
python distill_llm.py mode=worker device.worker_gpu=0

# Start student training (DDP)
torchrun --nproc_per_node=2 distill_llm.py mode=student
```

## API Reference

### setup_logging

```python
setup_logging(level: int | str = "INFO") -> None
```

Configure logging for all softs modules.

### Broker

```python
Broker(
    frontend_endpoint: str = "ipc:///tmp/softs_frontend.sock",
    backend_endpoint: str = "ipc:///tmp/softs_backend.sock",
    control_endpoint: str = "ipc:///tmp/softs_control.sock",
    control_pub_endpoint: str = "ipc:///tmp/softs_control_pub.sock",
)

broker.run()  # Blocking
broker.start()  # Non-blocking (background thread)
broker.stop()
broker.stats  # BrokerStats with metrics
```

### Worker

```python
Worker(
    generator_fn: Callable[[str, dict | None], bytes],  # model_id, model_cfg -> bytes
    slot_size: int,  # Expected bytes per sample
    backend_endpoint: str = ...,
    control_pub_endpoint: str = ...,
    worker_id: int | None = None,  # Defaults to PID
)

worker.run()  # Blocking
worker.start()  # Non-blocking
worker.stop()
worker.generation  # Current generation counter
worker.model_id  # Current model ID
```

### StudentClient

```python
StudentClient(
    student_rank: int,  # 0 = leader
    slot_count: int,  # Shared memory slots
    batch_config: BatchConfig,
    frontend_endpoint: str = ...,
    control_endpoint: str = ...,
    control_pub_endpoint: str = ...,
)

client.hello() -> dict  # Register with broker
client.set_model(model_id, model_cfg=None) -> int  # Set model (leader only), returns generation
client.request_sample(timeout_ms=1000) -> SampleRef | None
client.release_slot(slot_id)  # Return slot to pool
client.start_sub_listener()  # Listen for model changes (non-leader)
client.generation  # Current generation
client.generation_value  # multiprocessing.Value for sharing with dataset
client.close()
```

### DistillIterableDataset

```python
DistillIterableDataset(
    student_rank: int,
    generation_value: Value,  # From client.generation_value
    slot_count: int,
    batch_config: BatchConfig,
    frontend_endpoint: str = ...,
    max_retries: int = 10,
    retry_delay: float = 0.01,
)
```

Infinite `IterableDataset` yielding `dict[str, Tensor]`.

### BatchConfig / TensorSpec

```python
TensorSpec(name: str, shape: tuple[int, ...], dtype: str)
spec.nbytes  # Bytes for this tensor
spec.torch_dtype  # torch.dtype

BatchConfig(specs: list[TensorSpec])
config.nbytes() -> int
config.encode(**tensors) -> bytes
config.decode(data: bytes) -> dict[str, Tensor]
config.decode_single(data: bytes, name: str) -> Tensor
config.tensor_names -> list[str]
config.get_spec(name) -> TensorSpec
```

## Protocol Details

The broker uses ZeroMQ with four sockets:

| Socket | Type | Purpose |
|--------|------|---------|
| Frontend | ROUTER | Student requests (REQUEST, HELLO, STATS) |
| Backend | ROUTER | Worker communication (READY, WORK, DONE) |
| Control | ROUTER | Leader commands (SET_MODEL, STOP) |
| ControlPub | PUB | Broadcasts (MODEL changes, STOP) |

Commands:
- `HELLO`: Register student/worker
- `REQUEST`: Student requests a sample slot to be filled
- `READY`: Worker is available for work
- `WORK`: Broker assigns work to worker
- `DONE`: Worker completed writing to slot
- `COMPLETE`: Broker notifies student slot is ready
- `SET_MODEL`: Leader sets new model
- `STOP`: Shutdown workers

## Troubleshooting

### "Shared memory name too long" (macOS)

macOS limits shared memory names to 31 characters. The library uses short prefixes (`sl_`).

### Stale samples after model switch

The generation counter ensures stale samples are discarded. If you see stale data, ensure:
1. Dataset checks `generation_value` before yielding
2. You're using `client.generation_value` (shared with dataset)

### Worker not receiving work

Check:
1. Broker is running
2. Worker called `hello()` and is in main loop
3. Student has called `set_model()` (workers wait for a model)

### Memory not released

Call `client.close()` to properly unlink shared memory. Use context managers:

```python
with StudentClient(...) as client:
    # ...
# Automatically closes and unlinks
```

## License

MIT
