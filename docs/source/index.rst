Soft Labels Market
===================

**Async Soft Labels Generation with PyTorch**

``softs`` is a data pipeline for on-the-fly generation of (training) data.
It's built on a **Marketplace** pattern where a central :class:`~softs.Broker` routes orders from :class:`~softs.Client` to
a :class:`~softs.Supplier` that provides the product (the model id).  Suppliers generate information
and write it to a pluggable transfer :class:`~softs.Medium`.  Clients read the results 
from the delivery address when it's ready.

A key design principle is to prevent Client-Supplier dependencies. All coordination is handled by the broker.
More on the design arch in :doc:`architecture`.

Key Features
------------

- **Fault tolerant broker**: broker detects dead suppliers/clients, re-queues failed work.
- **Fault tolerant parties**: suppliers/clients can detect and reset on broker failure/disconnection.
- **Zero-copy when possible**: when data flows through shared memory
- **Model-aware routing**: clients order by ``product_id``, broker dispatches to matching suppliers
- **Cancel support**: clients can ``discard()`` all pending orders or ``cancel()`` individual ones
- **Pluggable mediums**: shared memory, filesystem (mmap), TCP - or extend ``Medium``
- **PyTorch integration**: ``SoftIterableDataset`` for seamless ``DataLoader`` use

Quick Example
-------------

**1. Start the broker:**

.. code-block:: python

    from softs import Broker, EndpointConfig

    broker = Broker(endpoints=EndpointConfig())
    broker.start()
    # ... broker.stop() on shutdown

**2. Start a supplier:**

.. code-block:: python

    import torch
    from softs import Supplier, ShmMedium, BatchConfig, TensorSpec, EndpointConfig

    config = BatchConfig([
        TensorSpec("x", (3, 224, 224), "float32"),
        TensorSpec("y", (1000,), "float32"),
    ])

    def generate(product_id: str) -> bytes:
        return config.encode(
            x=torch.randn(3, 224, 224),
            y=torch.randn(1000),
        )

    supplier = Supplier(
        generator_fn=generate,
        product_ids=["resnet"],
        endpoint=EndpointConfig().backend,
        medium_cls=ShmMedium,
        slot_size=config.nbytes(),
    )
    supplier.start()
    # ... supplier.stop() on shutdown

**3. Train:**

.. code-block:: python

    from softs import Client, ShmMedium, BatchConfig, TensorSpec, EndpointConfig

    config = BatchConfig([
        TensorSpec("x", (3, 224, 224), "float32"),
        TensorSpec("y", (1000,), "float32"),
    ])

    client = Client(
        endpoint=EndpointConfig().frontend,
        medium_cls=ShmMedium,
        slot_size=config.nbytes(),
        num_slots=8,
    )
    client.hello()

    for _ in range(100):
        slot = client.request_sample("resnet", timeout_ms=5000)
        if slot is None:
            continue
        sample = config.decode(client.medium.read(slot))
        client.release_slot(slot)
        # sample["x"].shape == (3, 224, 224)

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
