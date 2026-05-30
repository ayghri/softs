Soft Labels Package
===================

**Async Soft Labels Generation with PyTorch**

``softs`` is a single-machine data pipeline for on-the-fly generation of
training data.  A central **broker** routes requests from **clients** to
**workers** based on ``model_id``.  Workers generate data and write it to a
pluggable transfer medium.  Clients read the results with zero-copy access.

Key Features
------------

- **Model-aware routing**: clients request by ``model_id``, broker dispatches to matching workers
- **Zero-copy data plane**: data flows through shared memory (or filesystem, TCP)
- **Cancel support**: clients can ``discard()`` all pending requests or ``cancel()`` individual ones
- **Fault tolerant**: broker detects dead workers/clients, re-queues failed work, workers send ``GOODBYE`` on exit
- **Timeouts**: client and worker ``send_timeout_ms`` prevents hanging when broker is down
- **Pluggable mediums**: shared memory, filesystem (mmap), TCP — or extend ``Medium``
- **PyTorch integration**: ``DistillIterableDataset`` for seamless ``DataLoader`` use

Quick Example
-------------

**1. Start the broker:**

.. code-block:: python

    from softs import Broker, EndpointConfig
    Broker(endpoints=EndpointConfig()).run()

**2. Start a worker:**

.. code-block:: python

    import torch
    from softs import Worker, BatchConfig, TensorSpec, EndpointConfig

    config = BatchConfig([
        TensorSpec("x", (32, 3, 224, 224), "float32"),
        TensorSpec("y", (32, 1000), "float32"),
    ])

    def generate(model_id: str) -> bytes:
        return config.encode(
            x=torch.randn(32, 3, 224, 224),
            y=torch.randn(32, 1000),
        )

    Worker(
        generator_fn=generate,
        model_ids=["resnet"],
        slot_size=config.nbytes(),
        endpoints=EndpointConfig(),
    ).run()

**3. Train:**

.. code-block:: python

    from softs import Client, BatchConfig, TensorSpec, EndpointConfig

    config = BatchConfig([
        TensorSpec("x", (32, 3, 224, 224), "float32"),
        TensorSpec("y", (32, 1000), "float32"),
    ])

    client = Client(slot_count=8, batch_config=config, endpoints=EndpointConfig())
    client.hello()

    for _ in range(100):
        slot = client.request_sample("resnet", timeout_ms=5000)
        if slot is None:
            continue
        batch = config.decode(client.medium.read(slot))
        client.release_slot(slot)
        # batch["x"].shape == (32, 3, 224, 224)

    client.close()

Installation
------------

.. code-block:: bash

    pip install softs

.. toctree::
   :hidden:
   :maxdepth: 2

   quickstart
   architecture
   notebooks/basic_distillation
   notebooks/model_switching
   api/index
