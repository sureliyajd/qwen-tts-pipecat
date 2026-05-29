# Phase 1 — RoPE Analysis (Talker vs Megakernel)

**Status: RESOLVED. No CUDA kernel changes required.**

## The question

The megakernel was tuned for Qwen3-0.6B (text). The Qwen3-TTS **talker** decoder
has the same transformer shape, but its `config.json` rope differs:

```json
"rope_theta": 1000000,
"rope_scaling": { "interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default" }
```

vs the kernel, which assumes `rope_theta = 10000`, standard 1-D half-split RoPE.
The worry: does `interleaved` + `mrope_section` mean the kernel applies RoPE
incorrectly for the talker?

## Two different "interleaved" — don't confuse them

1. **Rotation pairing** — how head_dim elements are paired for the 2-D rotation:
   - *half-split / NeoX* (kernel): pair `(i, i + d/2)`.
   - *interleaved / GPT-J*: pair `(2i, 2i+1)`.
2. **Interleaved MRoPE** (what the config means) — how the **three** position
   axes (temporal/height/width) are laid out across the frequency vector before
   cos/sin is computed. This is the Qwen3-VL / Qwen2.5-Omni TMRoPE feature.

The config's `interleaved: true` refers to **#2**, not #1.

## What the transformers source shows

`rotate_half` and `apply_rotary_pos_emb` in the Qwen3-VL text path
(same family as the TTS talker) use **half-split rotation**:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

q_embed = (q * cos) + (rotate_half(q) * sin)
```

`apply_interleaved_mrope` only **assembles the frequency tensor** (which position
axis drives which frequency slot). It does **not** touch the rotation pairing:

```python
freqs_t = freqs[0]                       # temporal everywhere by default
for dim, offset in enumerate((1, 2), start=1):  # height, width
    length = mrope_section[dim] * 3
    idx = slice(offset, length, 3)
    freqs_t[..., idx] = freqs[dim, ..., idx]
```

So the **rotation math is identical to the kernel's**.

## Why MRoPE collapses to plain 1-D RoPE for the talker

The talker generates an audio/text token stream with **no image/video** input.
Per the Qwen2.5-Omni design: text uses only the temporal position; audio frames
each get a single temporal id; there are no height/width positions.

Therefore the three MRoPE axes all carry the **same scalar position** `p` for
every talker token. When `freqs[0] == freqs[1] == freqs[2]`, the
`apply_interleaved_mrope` reassembly is a **no-op** (it copies identical values
into themselves). The result is exactly:

```
freqs_t = p * inv_freq        # standard 1-D RoPE
```

## Conclusion / what actually has to change

| Item | Verdict |
|---|---|
| Rotation pairing (half-split) | already correct in kernel — **no change** |
| MRoPE / `interleaved` | **no-op** at talker decode — **no change** |
| `rope_theta` 10000 → 1000000 | rebuild `cos_table`/`sin_table` in **Python** (`model.py`), kernel untouched |

**Net: the only change is one constant when building the cos/sin tables.**
`inv_freq = 1.0 / (1_000_000.0 ** (arange(0, 128, 2) / 128))`.

This holds **only while the talker has no vision input** (true for TTS). If a
future multimodal path feeds images/video to this decoder, MRoPE would no longer
collapse and this analysis would not apply.

## Proof

`scripts/verify_rope.py` — pure CPU, no GPU, no weights. It:
1. shows `apply_interleaved_mrope` is identity when the 3 axes share a position,
2. shows the kernel's half-split index math == HF `rotate_half`,
3. asserts max abs diff < 1e-6 across random q/k and many positions.
