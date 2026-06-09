# softs

A broker-based, single-machine data pipeline for on-the-fly training-data generation in PyTorch (e.g. teacher-student distillation).

## Overview

`softs` provides a **data-agnostic** message-routing system:

- **Broker**: Routes orders between clients and suppliers. Knows nothing about the data.
- **Suppliers**: Generate samples on demand and write raw bytes into client-owned memory.
- **Clients**: Own memory slots, order samples by `product_id`, then read and decode the bytes.

The library only moves bytes; what those bytes represent is up to your application. Use `BatchConfig` for PyTorch tensor encoding/decoding.

## Key Features

- **Zero-copy transfer**: suppliers write directly into client shared memory
- **Async pipeline**: clients train while suppliers generate the next sample
- **Model switching**: change the requested `product_id` mid-training (e.g. layer-by-layer distillation)
- **Fault tolerance**: suppliers/clients can crash and restart independently; the broker re-queues in-flight work
- **Pluggable mediums**: shared memory (default), memory-mapped file, or TCP

## Architecture

```
+---------------------------------------------+
|                   BROKER                     |
|          (message router, data-agnostic)     |
|                                              |
|     Frontend (ROUTER)     Backend (ROUTER)   |
|          ^                      ^            |
+----------+----------------------+------------+
           |                      |
     +-----+------+        +------+------+
     |  Clients   |        |  Suppliers  |
     | (DEALER)   |        |  (DEALER)   |
     +-----+------+        +------+------+
           |                      |
           |   writes bytes into  |
           +----->  Medium  <-----+
              (shm / mmap / tcp)
```

Two ZMQ ROUTER sockets: a **frontend** for clients and a **backend** for suppliers. There is no separate control or pub/sub channel — model switching is driven entirely by the client (see below).

### Message Flow

1. **Client** creates a medium with one or more slots and sends `ORDER` with `{order_id, product_id, address, offset}`
2. **Broker** queues the order and assigns it to an available **supplier** via `WORK`
3. **Supplier** generates sample bytes with your `generator_fn(product_id)` and writes them directly into the medium
4. **Supplier** sends `DONE`; the broker sends `FULFILLED` to the client
5. **Client** reads the bytes from the medium, decodes tensors, and trains

### Model Switching

A client switches models by discarding pending orders and ordering with a new `product_id`. With the dataset wrapper:

```python
dataset.set_model("layer_5")   # subsequent samples are generated for layer_5
```

Internally this calls `client.discard()` and resumes requesting the new `product_id`. The switch is **fenced**: the client bumps a generation counter, cancels orders the broker has not yet dispatched, and drops any in-flight results from the previous model (their slots are reclaimed only once the supplier's write completes). So you never read a stale sample after a switch. There is no cross-process broadcast, though — coordinate switches across processes yourself (e.g. `torch.distributed.barrier()` for DDP).

## Installation

```bash
pip install softs
```

**Dependencies**: `pyzmq`, `msgpack`, `torch`, `numpy`, `pyyaml`

## Quick Start

### 1. Define your data format with BatchConfig

`BatchConfig` describes the tensors in **one sample** (no batch dimension):

```python
from softs import BatchConfig, TensorSpec

config = BatchConfig([
    TensorSpec("x", (3, 224, 224), "float32"),
    TensorSpec("y", (1000,), "float32"),
])
```

### 2. Start the broker

```python
from softs import Broker, EndpointConfig, setup_logging

setup_logging("INFO")
endpoints = EndpointConfig()
broker = Broker(endpoints=endpoints)
broker.start()       # background thread; broker.stop() to shut down
```

### 3. Start supplier(s)

```python
import torch
from softs import Supplier, ShmMedium

def generate_sample(product_id: str) -> bytes:
    x = torch.randn(3, 224, 224)
    y = torch.randn(1000)
    return config.encode(x=x, y=y)

supplier = Supplier(
    generator_fn=generate_sample,
    product_ids=["my_model"],
    endpoint=endpoints.backend,
    medium_cls=ShmMedium,
    slot_size=config.nbytes(),
)
supplier.start()     # background thread; supplier.stop() to shut down
```

### 4. Train

Either drive a `Client` directly:

```python
from softs import Client, ShmMedium

client = Client(
    endpoint=endpoints.frontend,
    medium_cls=ShmMedium,
    slot_size=config.nbytes(),
    num_slots=16,
)
client.hello()

for _ in range(100):
    slot = client.request_sample("my_model", timeout_ms=5000)
    if slot is None:
        continue
    sample = config.decode(client.medium.read(slot))
    client.release_slot(slot)
    # sample["x"].shape == (3, 224, 224)

client.close()
```

…or use the PyTorch dataset wrapper:

```python
from torch.utils.data import DataLoader
from softs import SoftIterableDataset, ShmMedium

dataset = SoftIterableDataset(
    model_id="my_model",
    endpoint=endpoints.frontend,
    batch_config=config,
    medium_cls=ShmMedium,
    num_slots=16,
)
loader = DataLoader(dataset, batch_size=32, num_workers=0)

for batch in loader:
    x, y = batch["x"], batch["y"]   # x.shape == (32, 3, 224, 224)
    # training loop...
    break
```

## BatchConfig API

`BatchConfig` describes tensors and handles encoding/decoding:

```python
from softs import BatchConfig, TensorSpec

config = BatchConfig([
    TensorSpec("hidden", (512, 768), "bfloat16"),
    TensorSpec("labels", (512,), "int64"),
])

config.nbytes()                      # size of one encoded sample, in bytes
data = config.encode(hidden=hidden_tensor, labels=label_tensor)
tensors = config.decode(data)        # -> {"hidden": ..., "labels": ...}
hidden = config.decode_single(data, "hidden")

config.tensor_names                  # ['hidden', 'labels']
config.get_spec("hidden")            # TensorSpec
```

**Supported dtypes**: `float64`, `float32`, `float16`, `bfloat16`, `int64`, `int32`, `int16`, `int8`, `uint8`, `bool`

### Loading from YAML

```python
config = BatchConfig.from_yaml("config.yaml")

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

By default `softs` uses **POSIX shared memory** for zero-copy transfer. The client owns the medium; suppliers attach to it by address and write into it.

- `ShmMedium` — POSIX shared memory (default)
- `FilesystemMedium` — memory-mapped file
- `TCPMedium` — TCP sockets; the client runs a server, suppliers connect

Select a medium by passing `medium_cls` to both the `Client`/dataset and the `Supplier`:

```python
from softs import Client, Supplier, FilesystemMedium

Supplier(..., medium_cls=FilesystemMedium)
Client(..., medium_cls=FilesystemMedium)
```

### Custom Mediums

Extend the `Medium` base class:

```python
from softs.market.mediums import Medium

class MyMedium(Medium):
    def __init__(self, address, slot_size, num_slots, create=False, **kwargs):
        super().__init__(address, slot_size, num_slots, create)
        # set up backing storage at `address`

    @classmethod
    def attach(cls, address: str) -> "MyMedium":
        return cls(address=address, slot_size=0, num_slots=0, create=False)

    def write(self, slot_offset: int, data: bytes) -> bool:
        ...   # return False if the resource is gone

    def read(self, slot_id: int) -> bytes:
        ...   # return slot_id * self.slot_size .. + self.slot_size

    def close(self) -> None: ...
    def unlink(self) -> None: ...
```

## API Reference

### setup_logging

```python
setup_logging(level: int | str = "INFO") -> None
```

### Broker

```python
Broker(
    endpoints: EndpointConfig,
    supplier_timeout: float = 60.0,
    client_timeout: float = 120.0,
    max_queue_per_product: int = 5000,
)

broker.start()          # non-blocking (background poll thread)
broker.stop()
broker.get_stats()      # -> BrokerStats
```

### Supplier

```python
Supplier(
    generator_fn: Callable[[str], bytes],   # product_id -> bytes
    product_ids: list[str],
    endpoint: str,                          # EndpointConfig.backend
    medium_cls: type[Medium],
    slot_size: int,
    send_timeout_ms: int = 10000,
)

supplier.start()
supplier.stop()
```

### Client

```python
Client(
    endpoint: str,                          # EndpointConfig.frontend
    medium_cls: type[Medium],
    slot_size: int,
    num_slots: int,
    address: str | None = None,
    send_timeout_ms: int = 5000,
)

client.hello() -> dict
client.request_sample(product_id, timeout_ms=1000) -> int | None   # slot id
client.request_slot(product_id) -> str | None                       # order id (async)
client.release_slot(slot_id)
client.discard() -> int                  # cancel all pending orders
client.cancel(order_id) -> bool          # cancel one order
client.get_stats() -> dict
client.close()
```

### SoftIterableDataset / SoftDataLoader

```python
SoftIterableDataset(
    model_id: str,
    endpoint: str,                          # EndpointConfig.frontend
    batch_config: BatchConfig,
    medium_cls: type[Medium],
    num_slots: int = 8,
    max_retries: int = 10,
    retry_delay: float = 0.01,
)
dataset.set_model(model_id)                # switch product_id
dataset.model_id                           # current product_id
```

`SoftDataLoader` takes the same arguments plus standard `DataLoader` kwargs and exposes `set_model(...)`.

### BatchConfig / TensorSpec

```python
TensorSpec(name: str, shape: tuple[int, ...], dtype: str)
spec.nbytes
spec.torch_dtype

BatchConfig(specs: list[TensorSpec])
config.nbytes() -> int
config.encode(**tensors) -> bytes
config.decode(data: bytes) -> dict[str, Tensor]
config.decode_single(data: bytes, name: str) -> Tensor
config.tensor_names -> list[str]
config.get_spec(name) -> TensorSpec
```

## Protocol Details

The broker uses ZeroMQ with two ROUTER sockets:

| Socket | Type | Purpose |
|--------|------|---------|
| Frontend | ROUTER | Client commands (`HELLO`, `ORDER`, `CANCEL`, `DISCARD`, `STATS`) |
| Backend | ROUTER | Supplier commands (`HELLO`, `READY`, `DONE`, `GOODBYE`) |

Broker-initiated messages: `WORK` (to a supplier) and `FULFILLED` (to a client). Payloads are msgpack-encoded `[command, payload]` frames.

## Fault Tolerance

- **Supplier dies**: broker detects via liveness timeout and re-queues in-flight work
- **Supplier generator fails**: reports `success=False`; broker re-queues to another supplier
- **Supplier exits gracefully**: sends `GOODBYE`; broker removes it immediately
- **Client dies**: broker cancels all its pending orders
- **Broker down**: client/supplier `send_timeout_ms` prevents an infinite hang

## License

MIT
