import re
from collections import defaultdict
from torch import nn
from typing import Optional
import torch

Tensor = torch.Tensor

_SPEC_RE = re.compile(r"(inputs|outputs)\[([^\]]*)\]")


def parse_io_spec(product_id: str) -> tuple[list[str], list[str]]:
    """Parse a capture spec string into ``(inputs, outputs)`` module-name lists.

    The spec names which modules to capture, e.g.::

        "inputs[model.layers.0]|outputs[model.layers.0]"
        "inputs[]|outputs[model.layers.0|model.layers.5]"
        "outputs[model.layers.0]"                      # inputs section optional

    Items inside the brackets are ``|``-separated; the separator *between* the
    ``inputs[...]`` and ``outputs[...]`` blocks is ignored (the regex matches
    each block independently), so the inner and outer ``|`` never clash.
    Whitespace and empty items are dropped, and a missing section yields ``[]``.

    Returns
    -------
    (inputs, outputs):
        Two lists of module names (either may be empty).
    """
    sections = {"inputs": [], "outputs": []}
    for kind, body in _SPEC_RE.findall(product_id):
        sections[kind] = [s.strip() for s in body.split("|") if s.strip()]
    return sections["inputs"], sections["outputs"]


def transfer_to_device(obj, device: Optional[torch.device]):
    """
    Recursively transfer tensors in nested structures to target device.

    Args:
        obj: Input object (tensor, list, dict, tuple, or other)
        device: Target device

    Returns:
        Object with same structure, tensors moved to device
    """
    if device is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {k: transfer_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [transfer_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        # Preserve namedtuple types
        if hasattr(obj, "_fields"):
            return type(obj)(*[transfer_to_device(v, device) for v in obj])
        return tuple(transfer_to_device(v, device) for v in obj)
    else:
        # Non-tensor, non-container types (int, float, None, etc.)
        return obj


class _StopForward(Exception):
    """Internal early-exit signal, raised once all requested captures are done.

    :meth:`ModelIOCatcher.run` catches it. If you call the model directly while
    ``early_exit=True`` it will surface here - route forwards through ``run``.
    """


class ModelIOCatcher:
    """Capture inputs and/or outputs of named submodules during forward passes.

    Hooks are installed once on every submodule (``attach_hooks``). Each capture
    *session* is then scoped with a ``with`` block that arms a chosen subset of
    modules and collects their activations into a buffer; outside the block the
    hooks stay attached but inert.

    **Construction.** ``model`` is the module to instrument -
    ``dict(model.named_modules())`` defines the valid target names (e.g.
    ``"model.layers.0"`` or ``"model.layers.0.mlp"``). ``device`` is where
    captured tensors (inputs and outputs) are moved via
    :func:`transfer_to_device`; pass ``None`` to leave them in place.

    **Selecting what to capture.** ``inputs`` and ``outputs`` are iterables of
    module **names** - the dotted keys from ``model.named_modules()``. They are
    independent; a name may appear in either, both, or neither:

    - a name in ``inputs`` captures the ``args``/``kwargs`` the module is called
      with, via a forward pre-hook (before the module runs);
    - a name in ``outputs`` captures whatever the module returns, via a forward
      hook (after the module and its whole subtree have run).

    **The buffer.** ``__enter__`` returns a buffer shaped
    ``buf[name][kind] -> list``::

        buf["model.layers.0"]["inputs"]   # list of {"args": ..., "kwargs": ...}
        buf["model.layers.0"]["outputs"]  # list of module return values

    One entry is appended per forward call, so running N forwards inside the same
    ``with`` accumulates N entries (e.g. to gather a batch of activations).

    **early_exit.** With ``early_exit=True`` a forward is cut short the instant
    every requested input/output has been captured: the last capturing hook
    raises an internal ``_StopForward`` to skip the rest of the network. Drive
    forwards through :meth:`run` (not the model directly) so that signal is
    caught and the budget re-seeded per call - then
    ``for batch in loader: catcher.run(**batch)`` runs every batch, each stopping
    after the deepest requested module. Output hooks fire in post-order (a parent
    fires after its children), so requesting the output of ``model.layers.0``
    stops after layer 0 finishes, not inside its ``mlp``. A cut-short forward
    returns ``None`` (read results from ``buf``); if a requested module never
    runs, no exit fires and the forward completes. With ``early_exit=False`` no
    stop signal is raised, so calling the model directly or via :meth:`run` is
    interchangeable.

    Example
    -------
    >>> catcher = ModelIOCatcher(model, torch.device("cuda"))
    >>>
    >>> # Full forwards: capture layer 0's input+output over the loader.
    >>> with catcher(inputs=["model.layers.0"],
    ...              outputs=["model.layers.0"]) as buf:
    ...     for batch in loader:
    ...         catcher.run(**batch)                 # or model(**batch) here
    >>> len(buf["model.layers.0"]["outputs"])        # one entry per batch
    >>> buf["model.layers.0"]["inputs"][0]["args"][0].shape   # layer 0 input
    >>>
    >>> # Only need layer 5's output? Skip layers 6.. on *every* batch:
    >>> with catcher.for_product("outputs[model.layers.5]", early_exit=True) as buf:
    ...     for batch in loader:
    ...         catcher.run(**batch)                 # returns None, cut short
    >>> hidden = buf["model.layers.5"]["outputs"]    # one tensor per batch

    Notes
    -----
    The model is **never modified** - :meth:`run` is a plain model call wrapped in
    a ``try/except``, so it composes with DDP, ``torch.compile``, etc. Capturing
    is scoped to the ``with`` block: ``__exit__`` disarms the flag sets, and the
    returned ``buf`` stays valid for reading afterwards. Hooks from
    ``attach_hooks`` stay registered until ``detach_hooks``, so every forward pays
    a cheap per-module membership check.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

        self.named_modules = dict(model.named_modules())

        self._handles = []

        self._in_flags = set()
        self._out_flags = set()

        self._io_buffer = defaultdict(lambda: defaultdict(list))

        self._pending = set()
        self._pending_template = set()
        self._early_exit = False

        self.attach_hooks()

    def _get_module_input_catcher(self, module_name, module):
        def input_hook(module, args, kwargs):
            if module_name in self._in_flags:
                self._io_buffer[module_name]["inputs"].append(
                    {
                        "args": transfer_to_device(args, self.device),
                        "kwargs": transfer_to_device(kwargs, self.device),
                    }
                )
                if self._early_exit:
                    self._pending.discard((module_name, "in"))
                    if not self._pending:
                        raise _StopForward()

        return module.register_forward_pre_hook(input_hook, with_kwargs=True)

    def _get_module_output_catcher(self, module_name, module):
        def output_hook(module, args, output):
            if module_name in self._out_flags:
                self._io_buffer[module_name]["outputs"].append(
                    transfer_to_device(output, self.device)
                )

                if self._early_exit:
                    self._pending.discard((module_name, "out"))
                    if not self._pending:
                        raise _StopForward()

        return module.register_forward_hook(output_hook)

    def attach_hooks(self):
        for name, module in self.named_modules.items():

            self._handles.append(self._get_module_input_catcher(name, module))
            self._handles.append(self._get_module_output_catcher(name, module))

    def detach_hooks(self):
        for hook in self._handles:
            hook.remove()
        self._handles = []

    def __call__(self, inputs=(), outputs=(), early_exit=False):
        """Arm a capture session and return ``self`` for use as a context manager.

        Parameters
        ----------
        inputs:
            Iterable of module names whose call inputs (``args``/``kwargs``) to
            capture. Names must be keys of ``model.named_modules()``.
        outputs:
            Iterable of module names whose return values to capture.
        early_exit:
            If True, stop the forward as soon as every requested input/output
            has been captured (skips downstream compute).

        Use as ``with catcher(inputs=[...], outputs=[...]) as buf:`` - the
        ``with`` target ``buf`` is the capture buffer (see the class docstring).
        Unknown module names raise ``KeyError``.
        """
        inputs, outputs = list(inputs), list(outputs)
        unknown = [n for n in (*inputs, *outputs) if n not in self.named_modules]
        if unknown:
            raise KeyError(
                f"unknown module name(s) {unknown}; names must be keys of "
                f"model.named_modules()"
            )
        self._pending_template = (
            {(n, "in") for n in inputs} | {(n, "out") for n in outputs}
            if early_exit
            else set()
        )
        self._pending = set(self._pending_template)
        self._in_flags = set(inputs)
        self._out_flags = set(outputs)
        self._early_exit = early_exit
        self._io_buffer = defaultdict(lambda: defaultdict(list))

        return self

    def for_product(self, product_id: str, early_exit: bool = True):
        """Arm a session from a ``product_id`` capture spec (see :func:`parse_io_spec`).

        ``product_id`` looks like ``"inputs[model.layers.0]|outputs[model.layers.0]"``.
        Returns ``self`` for use as ``with catcher.for_product(pid) as buf:``.
        Raises ``ValueError`` if the spec selects nothing and ``KeyError`` for
        unknown module names.
        """
        inputs, outputs = parse_io_spec(product_id)
        if not inputs and not outputs:
            raise ValueError(f"product_id {product_id!r} selects no modules")
        return self(inputs=inputs, outputs=outputs, early_exit=early_exit)

    def run(self, *args, **kwargs):
        """Run one forward under the active capture session.

        Use this instead of calling the model directly: it re-seeds the
        early-exit budget for this forward and absorbs the internal
        ``_StopForward`` signal, so ``for batch in loader: catcher.run(**batch)``
        runs every batch. Returns the model's output, or ``None`` if early exit
        cut the forward short. (With ``early_exit=False`` it is just the model
        call and always returns the real output.)

        The model is never modified - this is a plain call wrapped in a
        ``try/except``, so it composes with DDP, ``torch.compile``, etc.
        """
        self._pending = set(self._pending_template)
        try:
            return self.model(*args, **kwargs)
        except _StopForward:
            return None

    def __enter__(self):
        return self._io_buffer

    def __exit__(self, exc_type, exc, tb):
        self._in_flags = set()
        self._out_flags = set()
        return False
