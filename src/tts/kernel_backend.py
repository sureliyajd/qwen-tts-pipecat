"""
Phase 5 — run the talker trunk on the qwen_megakernel (RTX 5090 only).

Replaces `Qwen3TTSTalkerModel.forward` (the 28-layer body) with the megakernel, via
`src.tts.trunk.install_trunk_backend`. Everything else (text_projection, 16-group sum,
code_predictor, codec_head, sampling) stays in PyTorch — see docs/talker-decode.md.

How the kernel is driven (no kernel-source edits):
  - The kernel reads `embed_weight[token_id]` as the layer-0 input. We need an arbitrary
    1024-vector instead, so per token we set the Decoder's `embed_weight` to point at that
    vector (row 0) and pass `token_id=0` -> layer 0 reads our `inputs_embeds`.
  - We read `g_normalized` (final-normed hidden, Decoder._norm_out) as `last_hidden_state`
    and ignore the fused argmax LM head. The bundled `decode` op still runs that LM head
    over 151936 rows, so we hand it a dummy [151936,1024] buffer to avoid an OOB read
    (~311 MB, wasted ~0.3 ms/step; fine for correctness-first, optimize later).
  - The Decoder owns its KV cache + position counter. Prefill is the only multi-token
    call (T>1) -> we reset() then feed tokens sequentially. Gen steps are T==1.

Assumes batch size 1 (one stream per kernel instance) and a single prefill call.
"""

from __future__ import annotations

TEXT_VOCAB = 151936  # kernel LDG_VOCAB_SIZE (hardcoded); dummy LM-head row count
HIDDEN = 1024


class KernelTalkerTrunk:
    def __init__(self, talker_weights, device="cuda"):
        """`talker_weights` = output of src.tts.weights.load_talker_weights (has
        embed_weight, layer_weights[11x28], final_norm_weight, cos/sin @ theta 1e6)."""
        import torch
        from qwen_megakernel.model import Decoder

        self.device = device
        w = dict(talker_weights)
        # Dummy LM head so the bundled fused LM head doesn't read out of bounds.
        # Content irrelevant (we ignore the argmax output).
        w["lm_head_weight"] = torch.zeros(TEXT_VOCAB, HIDDEN, dtype=torch.bfloat16,
                                          device=device)
        self.dec = Decoder(weights=w, tokenizer=None, verbose=False)
        self._torch = torch

    def reset(self):
        self.dec.reset()

    def __call__(self, *args, inputs_embeds=None, cache_position=None,
                 past_key_values=None, **kwargs):
        """Drop-in for Qwen3TTSTalkerModel.forward. Returns BaseModelOutputWithPast."""
        torch = self._torch
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if inputs_embeds is None and args:
            inputs_embeds = args[0]
        ie = inputs_embeds[0]                     # [T,1024], batch 0
        T = ie.shape[0]

        # multi-token call == prefill -> fresh sequence
        if T > 1:
            self.dec.reset()

        outs = []
        for t in range(T):
            vec = ie[t].to(torch.bfloat16).contiguous().view(1, HIDDEN)
            self.dec._embed_weight = vec          # layer-0 input := our vector
            self.dec.step(0)                      # token_id=0 -> reads row 0 = vec
            outs.append(self.dec._norm_out.clone())   # final-normed hidden [1024] fp32

        hidden = torch.stack(outs, dim=0).unsqueeze(0).to(ie.dtype)  # [1,T,1024]
        return BaseModelOutputWithPast(last_hidden_state=hidden,
                                       past_key_values=past_key_values)


def install(model, talker_weights, device="cuda"):
    """Swap the talker trunk for the kernel. Returns (backend, restore_fn)."""
    from .trunk import install_trunk_backend
    backend = KernelTalkerTrunk(talker_weights, device=device)
    restore = install_trunk_backend(model, backend)
    return backend, restore
