"""
Phase 3 — the talker-trunk seam.

The megakernel replaces exactly one thing: `Qwen3TTSTalkerModel.forward`, i.e. the
28-layer decoder body that maps `inputs_embeds [B,T,1024]` -> `last_hidden_state
[B,T,1024]` (RMSNorm/QKV/RoPE/attn/O/MLP x28 + final norm). See docs/talker-decode.md.

This module provides:
  - `capture_trunk_io(model)`  -> records (inputs_embeds, last_hidden_state) per trunk
                                  call, for building golden pairs to test the kernel.
  - `install_trunk_backend(model, fn)` -> swap the trunk forward (Phase 5 kernel hook).
  - `KernelTrunkBackend` (stub) -> documents the kernel injection contract.

Everything stays compatible with HF GenerationMixin: we only touch `model.talker.model`.
torch is imported lazily so this file imports without torch installed.
"""

from __future__ import annotations


def _talker_trunk(model):
    """Return the Qwen3TTSTalkerModel (the 28-layer trunk) inside a loaded model.

    Handles both the transformers `Qwen3TTSForConditionalGeneration` and the
    qwen_tts inference wrapper (which holds the HF model on `.model` or similar).
    """
    for attr in ("talker", "model"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model
    # qwen_tts wrapper: dig one level
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "talker"):
        return inner.talker.model
    raise AttributeError("could not locate Qwen3TTSTalkerModel (.talker.model) on model")


class TrunkIO:
    """Accumulates trunk input/output tensors (CPU, fp32) for golden comparison."""

    def __init__(self):
        self.inputs = []   # list of [T,1024]
        self.outputs = []  # list of [T,1024]
        self.cache_pos = []

    def __len__(self):
        return len(self.inputs)


def capture_trunk_io(model) -> "tuple[TrunkIO, callable]":
    """Register a forward hook on the talker trunk; returns (TrunkIO, remove_fn).

    Records the trunk's `inputs_embeds` (from kwargs) and `last_hidden_state` per call.
    Does NOT change behavior — safe to run alongside a normal generate().
    """
    trunk = _talker_trunk(model)
    io = TrunkIO()

    def hook(_module, args, kwargs, output):
        ie = kwargs.get("inputs_embeds")
        if ie is None and args:
            ie = args[0]
        hs = getattr(output, "last_hidden_state", None)
        if hs is None and isinstance(output, (tuple, list)):
            hs = output[0]
        if ie is not None and hs is not None:
            io.inputs.append(ie.detach().float().cpu()[0])
            io.outputs.append(hs.detach().float().cpu()[0])
            cp = kwargs.get("cache_position")
            io.cache_pos.append(None if cp is None else cp.detach().cpu())
        return output

    handle = trunk.register_forward_hook(hook, with_kwargs=True)
    return io, handle.remove


def install_trunk_backend(model, forward_fn):
    """Replace the talker trunk's forward with `forward_fn(**kwargs) -> output`.

    `forward_fn` must accept the same kwargs as Qwen3TTSTalkerModel.forward and return
    an object with `.last_hidden_state` (and a valid `.past_key_values` if use_cache).
    Returns a restore() callable. Used in Phase 5 to route the trunk through the kernel.
    """
    trunk = _talker_trunk(model)
    original = trunk.forward
    trunk.forward = forward_fn  # type: ignore[assignment]

    def restore():
        trunk.forward = original

    return restore


class KernelTrunkBackend:
    """Phase-5 stub: run the talker trunk on the qwen_megakernel.

    Contract (see docs/talker-decode.md): single-token decode with an internal KV
    cache. For each token, inject the 1024-vector `inputs_embeds` via the embed-pointer
    trick (token_id=0, embed_weight := that row), run the persistent kernel, read
    `g_normalized` (final-normed hidden) as `last_hidden_state`. Prefill = N sequential
    single-token calls building the KV cache. Batch size 1 per kernel instance.

    NOTE: not implemented until Phase 5 (needs the RTX 5090). Kept here so the seam and
    its contract live next to the reference path.
    """

    def __init__(self, weights, meta, max_seq_len=2048):
        self.weights = weights
        self.meta = meta
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):  # pragma: no cover - Phase 5
        raise NotImplementedError(
            "KernelTrunkBackend runs only on RTX 5090 (Phase 5). "
            "Use capture_trunk_io + the HF trunk for CPU validation now."
        )
