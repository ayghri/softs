Architecture
============

Overview
--------

softs uses a **marketplace** architecture. Clients place orders (requests for a
concrete ``product_id`` string), suppliers register one or more **regex
patterns** describing what they can serve, and the broker dispatches each order
to a ready supplier whose pattern matches the order's ``product_id``. Data flows
through a pluggable transfer medium - the broker only routes the opaque
``(address, offset)`` location and never touches the payload.

The "products" are defined before hand via configs that form a "contract"
between clients and suppliers: defined from a single spec/file.

.. mermaid::

   graph LR
       C1[Client 1] & C2[Client 2] <-->|ORDER / FULFILLED| FE

       subgraph Broker
           FE[Frontend<br/>ROUTER] <--> Q[Per-product order queues<br/>+ supplier pools] <--> BE[Backend<br/>ROUTER]
       end

       BE <-->|WORK / DONE| W1[Supplier A<br/>products: v1] & W2[Supplier B<br/>products: v1, v2]

Broker
------

Two ZMQ ROUTER sockets: a frontend for clients and a backend for suppliers. A
single background poll thread services both. Orders are the single source of
truth, indexed for dispatch by:

- a per-product FIFO queue of pending orders,
- a single FIFO pool of ready suppliers (each carrying its compiled patterns),
- the in-flight order assigned to each busy supplier.

Client commands (frontend): ``HELLO``, ``ORDER``, ``CANCEL``, ``DISCARD``,
``STATS``. The broker pushes ``FULFILLED`` (or ``FAILED``, after
``max_order_attempts``) back to the client.

Supplier commands (backend): ``HELLO``, ``READY``, ``DONE``, ``GOODBYE``. The
broker pushes ``WORK`` to a supplier.

.. _routing-by-pattern:

Routing by pattern
~~~~~~~~~~~~~~~~~~~

A supplier's ``product_ids`` are **regex patterns**, not literal ids. On HELLO
the broker compiles them; to dispatch a queued order it scans the ready pool for
the first supplier whose pattern ``re.fullmatch``-es the order's concrete
``product_id``. A pattern with no regex metacharacters (``layer_3``) therefore
behaves as an exact match, while ``layer_\d+`` or ``.*`` lets one supplier serve
a whole family of ids without enumerating them. Because the ready pool holds each
supplier once, a supplier is dispatched a single order at a time and only
re-enters the pool when it sends ``READY`` again after ``DONE``.

Supplier
--------

1. Connects to the backend (DEALER), sends HELLO with its ``product_ids``
   (regex patterns it can serve)
2. Sends READY
3. Receives WORK with ``{order_id, product_id, address, offset}``
4. Calls ``generator_fn(product_id)`` -> bytes
5. Writes to the medium via ``medium_cls.attach(address).write(offset, data)``
6. Sends DONE with ``{order_id, success}``, then READY again
7. On failure: ``success=False`` -> broker re-queues to another supplier
8. On exit: sends GOODBYE -> broker removes it immediately

Client
------

1. Creates a transfer medium (one segment, multiple slots)
2. Connects to the frontend (DEALER), sends HELLO
3. Sends ORDER with ``{order_id, product_id, address, offset}``
4. Receives FULFILLED with ``{order_id}`` when the slot is filled
5. Reads from the medium, releases the slot
6. DISCARD cancels all pending orders, CANCEL cancels one

Message Flow
------------

.. mermaid::

   sequenceDiagram
       participant C as Client
       participant B as Broker
       participant S as Supplier

       S->>B: HELLO(product_ids)
       B->>S: {ok}
       S->>B: READY

       C->>B: HELLO
       B->>C: {ok, peer_id}

       C->>B: ORDER(order_id, product_id, address, offset)
       B->>C: {ok}
       B->>S: WORK(order_id, product_id, address, offset)
       Note over S: generate + write to medium
       S->>B: DONE(order_id, success=true)
       B->>C: FULFILLED(order_id)
       Note over C: read from medium

Cancel Flow
~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
       participant C as Client
       participant B as Broker

       C->>B: DISCARD
       Note over B: cancel client's QUEUED orders,<br/>let DISPATCHED ones finish
       B->>C: {ok, cancelled=N, cancelled_ids=[...]}

       C->>B: CANCEL(order_id)
       Note over B: cancel only if still QUEUED
       B->>C: {ok} or {ok:false, already dispatched}

Fencing model switches
~~~~~~~~~~~~~~~~~~~~~~~~

A switch must never serve a sample generated for the previous model. The client
holds a **generation** counter, bumped on every ``discard()``. Each order
records the generation it was placed in. On discard the broker cancels only
orders it has not yet dispatched (no supplier has touched their slots); orders
already dispatched are left to finish. The client keeps those slots quarantined
until their ``FULFILLED`` arrives - and because they belong to an old
generation, their data is dropped and the slot is freed rather than served. A
supplier write can therefore never land in a slot that has been reused for the
new model.

Supplier Failure
~~~~~~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
       participant C as Client
       participant B as Broker
       participant S1 as Supplier 1
       participant S2 as Supplier 2

       B->>S1: WORK(order_id)
       Note over S1: generator_fn raises!
       S1->>B: DONE(order_id, success=false)
       Note over B: re-queue to another supplier
       B->>S2: WORK(order_id)
       Note over S2: generates successfully
       S2->>B: DONE(order_id, success=true)
       B->>C: FULFILLED(order_id)

Fault Tolerance
---------------

- **Supplier disconnect**: broker removes it from product pools, re-queues in-flight work
- **Supplier failure**: ``success=False`` -> broker re-queues to another supplier
- **Supplier graceful exit**: GOODBYE -> broker handles immediately
- **Client disconnect**: broker cancels all its orders
- **Broker down**: ``send_timeout_ms`` returns an error instead of hanging
- **No shutdown ordering**: each component has its own ZMQ context with ``linger=0``

Transfer Mediums
----------------

The broker routes only the opaque ``(address, offset)`` - it never touches the
data.

.. mermaid::

   graph LR
       C[Client] -->|"ORDER(address, offset)"| B[Broker]
       B -->|"WORK(address, offset)"| S[Supplier]
       C -->|creates| M[Medium]
       S -->|"attach + write"| M
       M -->|read| C

- ``ShmMedium``: POSIX shared memory. Zero-copy. Default.
- ``FilesystemMedium``: memory-mapped file.
- ``TCPMedium``: TCP sockets. The client runs a server, suppliers connect.

Custom mediums extend the ``Medium`` base class: ``write()``, ``read()``,
``attach()``, ``close()``, ``unlink()``.

Capturing activations as products
---------------------------------

A common supplier is a *teacher* that serves a neural network's intermediate
activations as training samples. :class:`~softs.catcher.ModelIOCatcher` captures
the inputs and/or outputs of named submodules during a forward pass, and the
order's ``product_id`` selects what to capture.

Capture specs
~~~~~~~~~~~~~

A ``product_id`` is a spec string parsed by
:func:`~softs.catcher.parse_io_spec`::

    inputs[model.layers.0]|outputs[model.layers.0]
    outputs[model.layers.5]
    inputs[]|outputs[model.layers.0|model.layers.5]

The names are the dotted keys of ``model.named_modules()``. Items inside the
brackets are ``|``-separated; the ``|`` between the ``inputs[...]`` and
``outputs[...]`` blocks is matched independently, so the inner and outer
separators never clash.

Early exit
~~~~~~~~~~

With ``early_exit=True`` the catcher stops the forward the instant every
requested value is captured - so serving ``outputs[model.layers.0]`` only runs
the model *through layer 0* and skips the rest of the network. Forwards are
driven through :meth:`~softs.catcher.ModelIOCatcher.run` (not the model
directly), which catches the internal stop signal per call; the model itself is
never patched, so it composes with DDP and ``torch.compile``.

Teacher supplier
~~~~~~~~~~~~~~~~

Because the broker routes by :ref:`pattern <routing-by-pattern>`, one teacher
registers a single regex - e.g.
``r"inputs\[.*\]\|outputs\[.*\]"`` - and serves *any* layer spec a client asks
for, without enumerating them. The supplier's ``generator_fn`` parses the
``product_id``, arms the catcher, runs one forward, and encodes the captured
tensors to bytes::

    def generate(product_id: str) -> bytes:
        with catcher.for_product(product_id, early_exit=True) as buf:
            catcher.run(**next(batches))
        name = parse_io_spec(product_id)[1][0]
        return config.encode(
            x=buf[name]["inputs"][0]["args"][0][0].cpu(),
            y=buf[name]["outputs"][0][0].cpu(),
        )

See ``examples/distillation/qwen_layer0/`` for a minimal single-layer
distillation (broker/supplier/client, fixed layer, no switching), and
``examples/distillation/layerwise/`` for the all-layers version (one regex
pattern, ``early_exit``, and ``SoftDataLoader`` with per-layer ``set_model``
switching).
