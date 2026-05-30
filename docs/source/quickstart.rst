Quickstart
==========

Installation
------------

.. code-block:: bash

    pip install softs

Dependencies: Python 3.11+, PyTorch 2.0+, PyZMQ, msgpack, PyYAML.

Concepts
--------

Broker
~~~~~~

Routes requests from clients to workers based on ``model_id``.
Two ZMQ ROUTER sockets (frontend for clients, backend for workers).
Detects dead peers via liveness timeouts. Re-queues failed work.

Worker
~~~~~~

Registers ``model_ids`` it can serve. Receives work, calls your
``generator_fn(model_id) -> bytes``, writes to the client's medium.
Sends ``GOODBYE`` on clean exit. Has ``send_timeout_ms`` for broker outages.

Client
~~~~~~

Creates a medium, requests samples by ``model_id``, reads completed data.
``discard()`` cancels all pending, ``cancel(token)`` cancels one.
Has ``send_timeout_ms`` for broker outages. No ordering dependency on shutdown.

BatchConfig
~~~~~~~~~~~

Describes the tensor layout for one slot. Shapes include batch dimension::

    config = BatchConfig([
        TensorSpec("x", (B, 3, 224, 224), "float32"),
        TensorSpec("y", (B, 1000), "float32"),
    ])

Running
-------

**Broker:**

.. code-block:: python

    from softs import Broker, EndpointConfig
    Broker(endpoints=EndpointConfig()).run()

**Worker:**

.. code-block:: python

    import torch
    from softs import Worker, BatchConfig, TensorSpec, EndpointConfig

    config = BatchConfig([TensorSpec("x", (4,), "float32")])

    Worker(
        generator_fn=lambda mid: config.encode(x=torch.randn(4)),
        model_ids=["my_model"],
        slot_size=config.nbytes(),
        endpoints=EndpointConfig(),
    ).run()

**Client:**

.. code-block:: python

    from softs import Client, BatchConfig, TensorSpec, EndpointConfig

    config = BatchConfig([TensorSpec("x", (4,), "float32")])

    client = Client(slot_count=8, batch_config=config, endpoints=EndpointConfig())
    client.hello()

    slot = client.request_sample("my_model", timeout_ms=2000)
    tensors = config.decode(client.medium.read(slot))
    client.release_slot(slot)
    client.close()

Model Switching
---------------

Clients switch models by discarding old requests and requesting with a new
``model_id``.  For DDP sync, use ``torch.distributed.barrier()``:

.. code-block:: python

    client.discard()
    dist.barrier()

    for step in range(num_steps):
        slot = client.request_sample("layer_5", timeout_ms=5000)
        ...

Cancel & Discard
----------------

.. code-block:: python

    client.discard()              # cancel all pending requests
    client.cancel(token)          # cancel a specific request

Multiple Workers, Multiple Models
----------------------------------

.. code-block:: python

    # Worker A serves "v1"
    Worker(generator_fn=gen_v1, model_ids=["v1"], ...).run()

    # Worker B serves "v1" and "v2"
    Worker(generator_fn=gen_v2, model_ids=["v1", "v2"], ...).run()

    # Requests route to matching workers
    client.request_sample("v1")  # routed to A or B
    client.request_sample("v2")  # routed to B only

Fault Tolerance
---------------

- **Worker dies**: broker detects via liveness timeout, re-queues in-flight work
- **Worker exits gracefully**: sends GOODBYE, broker handles immediately
- **Worker generator fails**: sends ``success=False``, broker re-queues to another worker
- **Client dies**: broker cancels all its pending requests
- **Broker down**: client/worker ``send_timeout_ms`` prevents infinite hang
- **No shutdown ordering**: client, worker, broker can exit in any order

Using a Different Medium
------------------------

.. code-block:: python

    from softs import Worker, Client, FilesystemMedium

    Worker(..., medium_cls=FilesystemMedium).run()
    Client(..., medium_cls=FilesystemMedium)

Next Steps
----------

- :doc:`architecture` for internals
- :doc:`notebooks/basic_distillation` for a notebook walkthrough
- :doc:`api/broker` for full API reference
