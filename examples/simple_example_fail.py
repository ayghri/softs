#!/usr/bin/env python3
"""Failure-handling demo: what softs does when a supplier can't produce a sample.

This is the cautionary companion to ``simple_example.py``. Here the generator
*always* raises, so every order fails. It shows that softs surfaces and contains
the failure instead of hanging or failing silently:

  * the error is logged at each layer (medium / supplier / broker), not swallowed;
  * the broker retries a failing order up to ``max_order_attempts`` times;
  * then it gives up - it drops the order and tells the client (``FAILED``), so
    the client reclaims that slot instead of leaking it;
  * ``request_sample()`` returns ``None`` for the failed sample, so the training
    loop stays responsive and can simply skip ahead.

A medium write failure (wrong ``slot_size``, a detached shm segment, a bad
offset) surfaces the same way - see ``ShmMedium.write`` and
``Supplier._process_work``.

Run:
    python simple_example_fail.py
"""

import time

from softs import (
    Broker,
    Supplier,
    Client,
    ShmMedium,
    BatchConfig,
    TensorSpec,
    EndpointConfig,
    setup_logging,
)


def main():
    setup_logging("INFO")

    config = BatchConfig([TensorSpec("x", (8, 8), "float32")])
    endpoints = EndpointConfig(
        frontend="ipc:///tmp/softs_fail_fe.sock",
        backend="ipc:///tmp/softs_fail_be.sock",
    )

    def broken_generate(product_id: str) -> bytes:
        # Stand-in for a generator that blows up: OOM, a model error, bad data.
        raise RuntimeError("teacher model raised during generation")

    # max_order_attempts=3 keeps the demo short; the default is 5.
    broker = Broker(endpoints=endpoints, max_order_attempts=3)
    broker.start()
    time.sleep(0.2)

    supplier = Supplier(
        generator_fn=broken_generate,
        product_ids=["broken"],
        endpoint=endpoints.backend,
        medium_cls=ShmMedium,
        slot_size=config.nbytes(),
    )
    supplier.start()
    time.sleep(0.2)

    # Fewer slots (4) than the number of samples we request (8): if a failed
    # order leaked its slot, we would run out after 4 and the rest could never
    # be ordered. Reclaiming slots on FAILED is what keeps this working.
    client = Client(
        endpoint=endpoints.frontend,
        medium_cls=ShmMedium,
        slot_size=config.nbytes(),
        num_slots=4,
    )
    client.hello()

    num_samples = 8
    failed = 0
    print(f"\nRequesting {num_samples} samples from a 100%-failing supplier...\n")
    for i in range(num_samples):
        slot = client.request_sample("broken", timeout_ms=3000)
        if slot is None:
            failed += 1
            print(f"  sample {i}: order failed and was dropped - skipping")
            continue
        # A healthy run would decode here; we never reach this.
        client.medium.read(slot)
        client.release_slot(slot)

    # Reclaim any slots whose FAILED notice is still in flight.
    client.poll_completions(500)

    stats = broker.get_stats()
    print("\nSummary")
    print(f"  failed samples:     {failed}/{num_samples}")
    print(f"  broker completed:   {stats.total_completed} (none succeeded)")
    print(f"  broker pending:     {stats.pending_orders} (nothing stuck)")
    print(
        "\nThe loop stayed responsive and every slot was reclaimed despite a "
        "generator that never produced a sample."
    )

    client.close()
    supplier.stop()
    broker.stop()
    print("Done!")


if __name__ == "__main__":
    main()
