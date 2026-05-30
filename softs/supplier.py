"""Ray-based supplier actors and pool for async sample generation."""

import logging

import ray
import torch

logger = logging.getLogger(__name__)


@ray.remote
class _SupplierActor:
    """Internal Ray actor that wraps a generator."""

    def __init__(self, generator, *args, **kwargs):
        if isinstance(generator, type):
            self._gen = generator(*args, **kwargs)
        else:
            self._gen = generator

    def generate(self, product_id: str) -> dict[str, torch.Tensor]:
        return self._gen(product_id)

    def set_model(self, model_id: str) -> None:
        if hasattr(self._gen, "set_model"):
            self._gen.set_model(model_id)


class Supplier:
    """A GPU/CPU worker that generates samples via a Ray actor.

    Args:
        generator: A callable ``(product_id) -> dict[str, Tensor]``,
            or a **class** whose instances are callable. Classes are
            instantiated inside the actor with ``*args, **kwargs``.
            If the class has a ``set_model`` method, it can be called
            via ``supplier.set_model(model_id)`` to swap models.
        num_gpus: GPUs per supplier (0 for CPU-only).
        num_cpus: CPUs per supplier.
        *args, **kwargs: Forwarded to class constructor if generator is a class.

    Examples::

        class Teacher:
            def __init__(self, model_name):
                self.model = load_model(model_name).cuda()
            def __call__(self, product_id):
                with torch.no_grad():
                    return {"logits": self.model(get_batch()).logits.cpu()}
            def set_model(self, model_id):
                self.model = load_model(model_id).cuda()

        supplier = Supplier(Teacher, "Qwen/Qwen3-8B", num_gpus=1)
        supplier.set_model("Qwen/Qwen3-14B")  # hot-swap
    """

    def __init__(self, generator, *args, num_gpus: int = 0, num_cpus: int = 1, **kwargs):
        actor_cls = _SupplierActor.options(num_gpus=num_gpus, num_cpus=num_cpus)
        self._actor = actor_cls.remote(generator, *args, **kwargs)

    def generate(self, product_id: str) -> ray.ObjectRef:
        """Submit a generation request. Returns a Ray ObjectRef."""
        return self._actor.generate.remote(product_id)

    def set_model(self, model_id: str) -> ray.ObjectRef:
        """Signal this supplier to switch models. Returns ref to await completion."""
        return self._actor.set_model.remote(model_id)


class SupplierPool:
    """Manages multiple suppliers with async prefetching.

    Keeps ``prefetch`` requests in flight at all times. When one completes,
    it is returned and a new request is submitted to keep the pipeline full.

    Args:
        suppliers: List of Supplier instances.
        prefetch: Number of requests to keep in flight.
    """

    def __init__(self, suppliers: list[Supplier], prefetch: int = 4):
        self._suppliers = suppliers
        self._prefetch = prefetch
        self._pending: list[ray.ObjectRef] = []
        self._idx = 0

    def _submit(self, product_id: str) -> None:
        s = self._suppliers[self._idx % len(self._suppliers)]
        self._idx += 1
        self._pending.append(s.generate(product_id))

    def get(self, product_id: str, timeout: float | None = None) -> dict[str, torch.Tensor] | None:
        """Get the next completed sample. Submits new requests to maintain prefetch depth."""
        while len(self._pending) < self._prefetch:
            self._submit(product_id)
        ready, self._pending = ray.wait(self._pending, num_returns=1, timeout=timeout)
        if not ready:
            return None
        self._submit(product_id)
        return ray.get(ready[0])

    def set_model(self, model_id: str) -> None:
        """Switch all suppliers to a new model and drop pending results."""
        self.discard()
        ray.get([s.set_model(model_id) for s in self._suppliers])

    def set_model_distributed(self, model_id: str) -> None:
        """Switch model across all DDP ranks.

        Call from **every** rank. Rank 0 triggers the actual supplier
        switch; other ranks just discard stale prefetched data.
        Requires ``torch.distributed`` to be initialized.
        """
        import torch.distributed as dist

        dist.barrier()
        if dist.get_rank() == 0:
            self.set_model(model_id)
        else:
            self.discard()
        dist.barrier()

    def discard(self) -> int:
        """Drop all pending requests. Returns count dropped."""
        n = len(self._pending)
        self._pending.clear()
        return n
