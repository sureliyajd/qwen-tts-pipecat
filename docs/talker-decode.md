# Phase 3 — Talker decode mechanism (dual-track) RESOLVED

Source: `reference/qwen_tts/modeling_qwen3_tts.py` (QwenLM/Qwen3-TTS), classes
`Qwen3TTSForConditionalGeneration.generate` (prefill) and
`Qwen3TTSTalkerForConditionalGeneration.forward` (decode step).

## The seam the megakernel replaces

The kernel replaces **exactly** `Qwen3TTSTalkerModel.forward`:

```
inputs_embeds [B,T,1024]  ->  28 decoder layers (RMSNorm, QKV, RoPE, attn, O, MLP)
                          ->  final RMSNorm  ->  last_hidden_state [B,T,1024]
```

Everything else stays in PyTorch:
- text embedding + `text_projection` (2-layer MLP 2048->2048->1024)
- the 16-group embedding **sum** (dual-track input build)
- `code_predictor` (groups 1-15)
- `codec_head` (1024->3072 logits) + sampling
- `get_rope_index`, prefill prompt assembly, speaker conditioning, eos handling

## Dual-track input construction (per decode step)

`Qwen3TTSTalkerForConditionalGeneration.forward`, generate branch:

```python
last_id_hidden = codec_embedding(input_ids)              # group-0 token -> [B,1,1024]
predictor_result = code_predictor.generate(              # groups 1..15
    inputs_embeds=cat(past_hidden, last_id_hidden), max_new_tokens=15, ...)
codec_ids = cat(input_ids, predictor_result.sequences)   # 16 group ids for this frame
codec_hiddens = [last_id_hidden] + [code_predictor.codec_embedding[i](group_{i+1}) ...]
inputs_embeds = codec_hiddens.sum(1, keepdim=True)       # SUM of 16 group embeddings
if generation_step < trailing_text_hidden.shape[1]:
    inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step]   # text track
else:
    inputs_embeds = inputs_embeds + tts_pad_embed
# -> trunk(inputs_embeds) -> hidden -> codec_head(hidden) -> logits -> sample group-0
past_hidden = hidden[:, -1:, :]
```

So **dual-track = element-wise add** of the acoustic track (sum of 16 codebook-group
embeddings) and the text track (projected text hidden, or pad). Frame-synchronous at
12.5 Hz; one trunk call per audio frame.

## Prefill (one-time, before the loop)

`generate` assembles `talker_input_embeds` = role prefix + `tts_pad*(N-2)` + `tts_bos`
+ codec tag/speaker embeds + (text or speaker x-vector), and `trailing_text_hidden`
(text streamed in during generation). bos/eos/pad are TEXT-vocab tokens passed through
`text_projection`. Then `talker.generate(inputs_embeds=..., trailing_text_hidden=...,
tts_pad_embed=...)`.

## Output

`talker_codes [T,16]` per frame; stop when group-0 == `codec_eos_token_id` (2150).
Codes -> speech tokenizer (`speech_tokenizer/`, separate model) -> 24 kHz wav.

## Sampling (group 0 / main talker), from examples/test_model_12hz_base.py

`max_new_tokens=2048, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
repetition_penalty=1.05`. Sub-talker (code_predictor) uses `subtalker_*` kwargs.

## RoPE — Phase 1 re-confirmed against real talker code

- `Qwen3TTSTalkerRotaryEmbedding.forward`: `emb = cat(freqs, freqs); cos = emb.cos()`
  == kernel's `cos(freqs).repeat(1,2)`. theta=1e6, attention_scaling=1.0.
- `apply_multimodal_rotary_pos_emb`: rotation is `q*cos + rotate_half(q)*sin`
  (half-split) — identical to kernel.
- `get_rope_index`: `position_ids = cumsum(mask)-1` then `expand(3,-1,-1)` -> all 3
  MRoPE axes identical (no vision) -> interleaved reassembly is identity.

=> Kernel RoPE is exactly correct; only theta differs (handled in weights.py). See
`docs/rope-analysis.md` + `scripts/verify_rope.py`.

## Kernel integration plan (Phase 5)

The kernel takes a token id and does `embed_weight[token_id]` for the layer-0 input.
The talker input is an arbitrary 1024-vector (the dual-track sum), not a token lookup.
Injection trick (no kernel edit): per step set `embed_weight` to point at the
[1,1024] `inputs_embeds` vector and pass `token_id=0`, so layer 0 reads our vector.
Read `g_normalized` (final-normed hidden) for `codec_head`; ignore the fused argmax
LM head. Open item: the bundled `decode` op still launches the 151936-vocab LM head —
either provide a dummy [151936,1024] buffer (~311MB, wasteful but unmodified) or add a
trunk-only launch path. Prefill = run kernel token-by-token to build KV (single-token
kernel); batch size 1 per kernel instance.
