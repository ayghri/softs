Quickstart
==========

Installation
------------

.. code-block:: bash

    pip install softs

Dependencies: Python 3.11+, PyTorch 2.9+, PyZMQ, msgpack, PyYAML.

Concepts
--------

Broker
~~~~~~

Routes orders from clients to suppliers based on ``product_id`` (the model
id). Two ZMQ ROUTER sockets: a frontend for clients and a backend for
suppliers. Detects dead peers via liveness timeouts and re-queues in-flight
work. Started with ``start()`` (spawns a background poll thread) and stopped
with ``stop()``.

Supplier
~~~~~~~~

Registers the ``product_ids`` it can serve. On each unit of work it calls your
``generator_fn(product_id) -> bytes`` and writes the bytes into the client's
medium. Sends ``GOODBYE`` on clean exit. ``send_timeout_ms`` bounds waits when
the broker is unreachable.

Client
~~~~~~

Creates a medium, requests samples by ``product_id``, and reads completed data
out of the medium. ``discard()`` cancels all pending orders, ``cancel(order_id)``
cancels one. ``send_timeout_ms`` bounds waits when the broker is unreachable.

BatchConfig
~~~~~~~~~~~

Describes the tensor layout of **one sample** (no batch dimension - the
DataLoader adds that). Each slot in the medium holds exactly one encoded
sample::

    from softs import BatchConfig, TensorSpec

    config = BatchConfig([
        TensorSpec("x", (3, 224, 224), "float32"),
        TensorSpec("y", (1000,), "float32"),
    ])

``config.nbytes()`` is the size of one encoded sample (the medium ``slot_size``).

Running
-------

The broker, supplier, and client each run on their own ZMQ context and can live
in separate processes or, as below, in a single process for a self-contained
example. ``Broker.start()`` and ``Supplier.start()`` spawn background threads.

**Broker:**

.. code-block:: python

    from softs import Broker, EndpointConfig

    endpoints = EndpointConfig()  # ipc:///tmp/softs_{frontend,backend}.sock
    broker = Broker(endpoints=endpoints)
    broker.start()
    # ... later ...
    broker.stop()

**Supplier:**

.. code-block:: python

    import torch
    from softs import Supplier, ShmMedium, BatchConfig, TensorSpec, EndpointConfig

    config = BatchConfig([TensorSpec("x", (4,), "float32")])
    endpoints = EndpointConfig()

    def generate(product_id: str) -> bytes:
        return config.encode(x=torch.randn(4))

    supplier = Supplier(
        generator_fn=generate,
        product_ids=["my_model"],
        endpoint=endpoints.backend,
        medium_cls=ShmMedium,
        slot_size=config.nbytes(),
    )
    supplier.start()
    # ... later ...
    supplier.stop()

**Client:**

.. code-block:: python

    from softs import Client, ShmMedium, BatchConfig, TensorSpec, EndpointConfig

    config = BatchConfig([TensorSpec("x", (4,), "float32")])
    endpoints = EndpointConfig()

    client = Client(
        endpoint=endpoints.frontend,
        medium_cls=ShmMedium,
        slot_size=config.nbytes(),
        num_slots=8,
    )
    client.hello()

    slot = client.request_sample("my_model", timeout_ms=2000)
    tensors = config.decode(client.medium.read(slot))
    client.release_slot(slot)
    client.close()

PyTorch DataLoader
------------------

``SoftIterableDataset`` is an infinite dataset that pulls samples from the
broker; wrap it in a standard ``DataLoader`` (the ``batch_size`` adds the batch
dimension):

.. code-block:: python

    from torch.utils.data import DataLoader
    from softs import SoftIterableDataset, ShmMedium

    dataset = SoftIterableDataset(
        model_id="my_model",
        endpoint=endpoints.frontend,
        batch_config=config,
        medium_cls=ShmMedium,
        num_slots=8,
    )
    loader = DataLoader(dataset, batch_size=32, num_workers=0)

    for batch in loader:
        x = batch["x"]  # shape (32, 4)
        ...

Model Switching
---------------

Switch models by calling ``set_model`` on the dataset (or ``discard`` +
re-request on a raw client). Pending work for the old model is dropped:

.. code-block:: python

    dataset.set_model("my_model_v2")  # subsequent samples use the new model

For DDP, gate the switch with a barrier so all ranks change together:

.. code-block:: python

    import torch.distributed as dist

    dataset.set_model("my_model_v2")
    dist.barrier()

Cancel & Discard
----------------

.. code-block:: python

    order_id = client.request_slot("my_model")  # returns the order id
    client.cancel(order_id)                      # cancel that one order
    client.discard()                             # cancel all pending orders

Multiple Suppliers, Multiple Models
-----------------------------------

.. code-block:: python

    # Supplier A serves "v1"
    Supplier(generator_fn=gen_v1, product_ids=["v1"], ...).start()

    # Supplier B serves "v1" and "v2"
    Supplier(generator_fn=gen_v2, product_ids=["v1", "v2"], ...).start()

    # Orders route to a matching supplier
    client.request_sample("v1")  # served by A or B
    client.request_sample("v2")  # served by B only

Fault Tolerance
---------------

- **Supplier dies**: broker detects via liveness timeout, re-queues in-flight work
- **Supplier exits gracefully**: sends GOODBYE, broker removes it immediately
- **Supplier generator fails**: reports ``success=False``, broker re-queues to another supplier
- **Client dies**: broker cancels all its pending orders
- **Broker down**: client/supplier ``send_timeout_ms`` prevents an infinite hang
- **No shutdown ordering**: each component has its own ZMQ context with ``linger=0``

Using a Different Medium
------------------------

.. code-block:: python

    from softs import Supplier, Client, FilesystemMedium

    Supplier(..., medium_cls=FilesystemMedium)
    Client(..., medium_cls=FilesystemMedium)

``ShmMedium`` (POSIX shared memory) is the default; ``FilesystemMedium``
(mmap'd file) and ``TCPMedium`` (client runs a server, suppliers connect) are
also available.

Next Steps
----------

- :doc:`architecture` for internals
- :doc:`api/broker` for the full API reference
