# PROGRESS — Qwen3-TTS megakernel + Pipecat

Single source of truth for project status. Updated 2026-05-30.

Phases 1-4 done and validated on free hardware (CPU + Colab T4). Phase-5 code written;
runs only on the RTX 5090. Nothing committed yet (see Git status).

---

## Model facts (confirmed)

- Real model: **`Qwen/Qwen3-TTS-12Hz-0.6B-Base`** (NOT `Qwen/Qwen3-TTS`). Native
  transformers (`model_type: qwen3_tts`), single `model.safetensors` (2.51 GB).
- 3 stages: **Talker LM** (28L, 1024 hidden — kernel target) -> **Code Predictor**
  (5L, groups 1-15, torch) -> **Vocoder/speech tokenizer** (causal, 24 kHz, torch).
- Talker trunk shape == Qwen3-0.6B exactly. Differs only: vocab 3072 (codec), rope_theta
  1e6, MRoPE-interleaved (collapses, see below). Audio 24 kHz, 12.5 Hz frames,
  1920 samples/frame, 16 codebook groups, group-0 EOS = 2150.
- **Base = voice-clone only** (`get_supported_speakers() == []`); use
  `generate_voice_clone(text, language, ref_audio, ref_text, x_vector_only_mode=False)`.
  Languages lowercase (`english`).
- Attribute paths: `tts`=`Qwen3TTSModel` wrapper; `hf=tts.model`
  (`Qwen3TTSForConditionalGeneration`); `hf.talker` / `hf.talker.model` (trunk);
  `hf.speech_tokenizer.model.decoder` (vocoder); spf via
  `hf.speech_tokenizer.get_decode_upsample_rate()` = 1920.
- Env pins (Colab): **transformers==4.57.3** + **huggingface_hub<1.0**. Newer transformers
  breaks: `check_model_inputs() missing 1 required positional argument: 'func'`. Do NOT
  `-U transformers`; restart runtime after re-pinning.

---

## Phase status

### Phase 1 — RoPE  ✅ DONE (proven)
Kernel RoPE needs **no CUDA change**, only `theta=1e6` when building cos/sin.
- `interleaved=true` = MRoPE freq assembly, not GPT-J pairing. Rotation is half-split
  `rotate_half` (= kernel). `get_rope_index` expands 3 equal axes (no vision) -> MRoPE is
  identity. cos/sin built as `cat(freqs,freqs)` (= kernel `repeat(1,2)`).
- Verified against the real talker code AND numerically: `scripts/verify_rope.py`
  (pure stdlib) -> exact 0.0 diff over 2048 positions.
- Docs: `docs/rope-analysis.md`.

### Phase 2 — Weight loader  ✅ DONE (confirmed vs real weights)
`src/tts/weights.py` — introspective (no hardcoded key strings): autodetects the talker
prefix by structure+shape; stdlib safetensors-header inspector (no torch/GPU).
- Confirmed real keys: trunk `talker.model.layers.N.*` (matches kernel 11-tensor/layer);
  `talker.model.codec_embedding.weight` [3072,1024]; `talker.model.norm.weight`;
  `talker.codec_head.weight` [3072,1024]; `talker.model.text_embedding.weight`
  [151936,2048]; `talker.text_projection.linear_fc1/fc2.{weight,bias}` (2048->2048->1024).
- `scripts/download_model.sh` (fixed: uv/pip/python3-m-pip fallback; dropped `[cli]` extra;
  split `--exclude` flags; needs `HF_HUB_DISABLE_XET=1` to avoid xet stalls).

### Phase 3 — Decode / dual-track  ✅ DONE (resolved from source)
Seam the kernel replaces = **`Qwen3TTSTalkerModel.forward`** only. Dual-track =
element-wise ADD per step: `inputs_embeds = sum(16 group embeds) + trailing_text_hidden`
(or `tts_pad_embed`). group-0 from `codec_head` sample; groups 1-15 from
`code_predictor.generate`. Frame-sync 12.5 Hz, one trunk call/frame.
- `src/tts/trunk.py` (seam: `capture_trunk_io`, `install_trunk_backend`),
  `scripts/capture_golden.py`. Docs: `docs/talker-decode.md`.
- Sampling (group0): do_sample, top_k=50, top_p=1.0, temp=0.9, rep_penalty=1.05.

### Phase 4 — Code predictor + vocoder + streaming  ✅ DONE (unit-tested + Colab)
Vocoder is fully causal with native `chunked_decode(chunk_size, left_context_size)`.
`src/tts/stream.py`: `AudioFrameStreamer` (forward hook -> per-frame 16 codes),
`StreamingVocoder` (incremental chunked decode), `stream_tts_pcm` / `astream_tts_pcm`.
Bypasses upstream's full-utterance buffering. Docs: `docs/streaming.md`.

### Colab validation  ✅ PASS (free, no kernel)
`docs/colab.md` + `scripts/colab_validate.py`. Three checks on T4:
- **[A]** greedy `generate_voice_clone` -> `A_reference.wav`, 46080 samp @ 24 kHz (~1.9 s),
  25 trunk calls (1 prefill + 24 gen), 24 frames. Audibly correct ("hello from the
  megakernel") — confirmed by user.
- **[B]** `golden_trunk.npz` — per-call talker `inputs_embeds` + `last_hidden_state` +
  codes. **The Phase-5 kernel test vectors.** (Saved in user's ~/Downloads.)
- **[C]** streamed vs full decode (same greedy codes): **fp32 max|diff| = 1.6e-6, lengths
  equal -> PASS.** (bf16 gives ~6.8e-3 = rounding from different reduction order, not a
  bug — hence `--fp32_check` default on.)

### Phase 5 — Kernel swap + server  🔜 CODE WRITTEN, NEEDS RTX 5090
- `src/tts/kernel_backend.py` — `KernelTalkerTrunk`: drives the megakernel `Decoder`,
  injects `inputs_embeds` via `dec._embed_weight=vec` + `step(0)`, reads `dec._norm_out`
  as hidden, dummy [151936,1024] LM-head buffer to avoid OOB, resets on T>1 prefill, bs=1.
  `install(model, weights)` swaps the trunk. No kernel-source edits.
- `scripts/verify_kernel_trunk.py` — FIRST GPU step: replay `golden_trunk.npz` through the
  kernel, compare hidden. **Use bf16-realistic tolerance (~1e-2 / mean<0.05), not 1e-3.**
- `docs/phase5-runbook.md` — Vast.ai setup + order (verify -> e2e audio -> server).
- Pending: `scripts/setup.sh` (still a stub), `src/server/main.py` (FastAPI/WS, build
  after kernel verified).

### Phases 6-8 — Pipecat service / full pipeline / benchmarks  ⬜ NOT STARTED
`astream_tts_pcm` is the hook for the Pipecat `TTSService` (Phase 6). Benchmark targets:
TTFC<60ms (Qwen's own first-packet = 97ms, so this is a stretch), RTF<0.15, ~1000 tok/s.

---

## Git status (nothing committed)
Changed/created, not committed:
- src/tts/{weights,trunk,stream,kernel_backend}.py
- scripts/{verify_rope,capture_golden,colab_validate,verify_kernel_trunk}.py,
  download_model.sh
- docs/{rope-analysis,talker-decode,streaming,colab,phase5-runbook,PROGRESS}.md
- CLAUDE.md (status), .gitignore (reference/)
Local only (gitignored / not in repo): `reference/` (vendored Qwen source),
`models/` (2.51 GB weights), `golden_trunk.npz` (in ~/Downloads).

## Immediate next actions
1. (optional) Fill `scripts/setup.sh` so the GPU box is one-command provision.
2. Commit the working tree (user's call).
3. Rent RTX 5090 -> `verify_kernel_trunk.py` against `golden_trunk.npz` (PASS gate).
