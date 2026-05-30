Architecture
============

Overview
--------

softs uses a **marketplace** architecture. Clients submit orders
(requests with a ``model_id``), workers serve orders for models they've
registered. The broker matches orders to workers. Data flows through a
pluggable transfer medium — the broker never touches it.

.. mermaid::

   graph LR
       C1[Client 1] & C2[Client 2] <-->|REQUEST / COMPLETE| FE

       subgraph Broker
           FE[Frontend<br/>ROUTER] <--> Q[Per-model queues<br/>+ worker pools] <--> BE[Backend<br/>ROUTER]
       end

       BE <-->|WORK / DONE| W1[Worker A<br/>models: v1] & W2[Worker B<br/>models: v1, v2]

Broker
------

Two ZMQ ROUTER sockets: frontend (clients) and backend (workers).

Data structures:

- ``_request_queues[model_id]`` — per-model FIFO of pending requests
- ``_available_workers[model_id]`` — per-model pool of idle workers
- ``_worker_models[identity]`` — which models each worker serves

Worker
------

1. Connects to backend (DEALER), sends HELLO with ``model_ids``
2. Sends READY
3. Receives WORK with ``{token, model_id, buf_name, slot_offset}``
4. Calls ``generator_fn(model_id)`` → bytes
5. Writes to medium via ``medium_cls.attach(buf_name).write(offset, data)``
6. Sends DONE with ``{token, success}``
7. On failure: ``success=False`` → broker re-queues to another worker
8. On exit: sends GOODBYE → broker removes immediately

Client
------

1. Creates a transfer medium (one segment, multiple slots)
2. Connects to frontend (DEALER), sends HELLO
3. Sends REQUEST with ``{token, model_id, buf_name, slot_offset}``
4. Receives COMPLETE with ``{token}`` when slot is filled
5. Reads from medium, releases slot
6. DISCARD cancels all pending, CANCEL cancels one

Message Flow
------------

.. mermaid::

   sequenceDiagram
       participant C as Client
       participant B as Broker
       participant W as Worker

       W->>B: HELLO(model_ids)
       B->>W: {ok}
       W->>B: READY

       C->>B: HELLO
       B->>C: {ok}

       C->>B: REQUEST(model_id, buf_name, offset)
       B->>C: {ok}
       B->>W: WORK(model_id, buf_name, offset)
       Note over W: generate + write to medium
       W->>B: DONE(token, success=true)
       B->>C: COMPLETE(token)
       Note over C: read from medium

Cancel Flow
~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
       participant C as Client
       participant B as Broker

       C->>B: DISCARD
       Note over B: remove all client's<br/>queued + pending work
       B->>C: {ok, cancelled=N}

       C->>B: CANCEL(token)
       Note over B: remove specific token
       B->>C: {ok}

Worker Failure
~~~~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
       participant C as Client
       participant B as Broker
       participant W1 as Worker 1
       participant W2 as Worker 2

       B->>W1: WORK(token)
       Note over W1: generator_fn raises!
       W1->>B: DONE(token, success=false)
       Note over B: re-queue to another worker
       B->>W2: WORK(token)
       Note over W2: generates successfully
       W2->>B: DONE(token, success=true)
       B->>C: COMPLETE(token)

Fault Tolerance
---------------

- **Worker disconnect**: broker removes from model pools, re-queues in-flight work
- **Worker failure**: ``success=False`` → broker re-queues to another worker
- **Worker graceful exit**: GOODBYE → broker handles immediately
- **Client disconnect**: broker cancels all its tokens
- **Broker down**: ``send_timeout_ms`` returns error instead of hanging
- **No shutdown ordering**: each component has its own ZMQ context with ``linger=0``

Transfer Mediums
----------------

The broker routes opaque ``(buf_name, slot_offset)`` — never touches data.

.. mermaid::

   graph LR
       C[Client] -->|"REQUEST(buf_name, offset)"| B[Broker]
       B -->|"WORK(buf_name, offset)"| W[Worker]
       C -->|creates| M[Medium]
       W -->|"attach + write"| M
       M -->|read| C

- ``ShmMedium``: POSIX shared memory. Zero-copy. Default.
- ``FilesystemMedium``: memory-mapped file.
- ``TCPMedium``: TCP sockets. Client runs server, workers connect.

Custom mediums extend ``Medium`` base class: ``write()``, ``read()``,
``attach()``, ``close()``, ``unlink()``.
