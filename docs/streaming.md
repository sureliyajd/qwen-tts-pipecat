# Phase 4 — code predictor, vocoder, streaming

## What's reused (not reimplemented)
- **Code predictor**: runs inside `Qwen3TTSTalkerForConditionalGeneration.forward`
  (groups 1-15). PyTorch. We do not touch it.
- **Vocoder / speech tokenizer decoder**: `Qwen3TTSTokenizerV2Decoder`. Fully causal
  (CausalConv + CausalTransConv + causal transformer + Snake/ConvNeXt upsample). Ships
  `chunked_decode(codes, chunk_size=300, left_context_size=25)` — re-decodes a small
  left context each chunk and trims `context * total_upsample` output samples. Causal =>
  trimmed chunks concatenate seamlessly. `decode_upsample_rate` = samples/frame.

## Numbers
24 kHz mono. 12.5 Hz frames => `SAMPLES_PER_FRAME = 24000/12.5 = 1920`. 16 codebook
groups/frame. Group-0 EOS = `codec_eos_token_id = 2150` (stop condition).

## Streaming design (src/tts/stream.py)
Upstream `generate_custom_voice` buffers the full utterance then decodes once — not
allowed here. Instead:

1. `AudioFrameStreamer` — forward hook on the talker CG. Each decode step the talker
   returns `hidden_states=(layer_hiddens, codec_ids)`; we push the `[16]` frame onto a
   queue as it is produced. Frame-by-frame, no full buffer.
2. `StreamingVocoder` — incremental `chunked_decode`: buffer frames, once `chunk_frames`
   are ready decode `[left_context + chunk]`, trim context samples, emit int16 PCM.
   Verified (torch-free) to tile frames exactly once (no gaps/dupes/reorder).
3. `stream_tts_pcm(model, text=...)` — runs `generate_custom_voice` in a worker thread,
   bridges frames -> vocoder, yields PCM. `astream_tts_pcm` = async (for Pipecat).

## Latency / quality knobs
- `chunk_frames`: small => lower first-audio latency, more decode calls. 1 frame = 80 ms
  of audio.
- `left_context`: bigger => better chunk-boundary quality, more recompute per chunk
  (RTF cost). First chunk has no context (no added latency). Defaults 6 / 25; tune in
  Phase 8 against the TTFC/RTF targets.

## Status
Code written + bookkeeping unit-tested offline. Running end-to-end needs torch +
qwen_tts + the model (Colab/GPU) — same as Phase 3. The talker trunk here is still HF;
Phase 5 swaps it for the kernel via `src/tts/trunk.install_trunk_backend`.
