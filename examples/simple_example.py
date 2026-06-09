#!/usr/bin/env python3
"""Self-contained softs demo: broker + suppliers + client in one process.

Run:
    python simple_example.py
"""

import time

import torch

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

    # One sample = two (64, 128) float32 tensors.
    config = BatchConfig(
        [
            TensorSpec("x", (64, 128), "float32"),
            TensorSpec("y", (64, 128), "float32"),
        ]
    )
    endpoints = EndpointConfig(
        frontend="ipc:///tmp/softs_simple_fe.sock",
        backend="ipc:///tmp/softs_simple_be.sock",
    )

    def generate(product_id: str) -> bytes:
        return config.encode(x=torch.randn(64, 128), y=torch.randn(64, 128))

    broker = Broker(endpoints=endpoints)
    broker.start()
    time.sleep(0.2)

    # Two suppliers serving the same product.
    suppliers = [
        Supplier(
            generator_fn=generate,
            product_ids=["random_gen"],
            endpoint=endpoints.backend,
            medium_cls=ShmMedium,
            slot_size=config.nbytes(),
        )
        for _ in range(2)
    ]
    for s in suppliers:
        s.start()
    time.sleep(0.2)

    client = Client(
        endpoint=endpoints.frontend,
        medium_cls=ShmMedium,
        slot_size=config.nbytes(),
        num_slots=8,
    )
    client.hello()

    num_samples = 50
    for i in range(num_samples):
        slot = client.request_sample("random_gen", timeout_ms=1000)
        if slot is None:
            print(f"Timeout at sample {i}")
            continue
        batch = config.decode(client.medium.read(slot))
        client.release_slot(slot)
        if i == 0:
            print(f"Shapes: {[(k, tuple(v.shape)) for k, v in batch.items()]}")
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_samples}")

    print("Stats:", broker.get_stats())

    client.close()
    for s in suppliers:
        s.stop()
    broker.stop()
    print("Done!")


if __name__ == "__main__":
    main()
