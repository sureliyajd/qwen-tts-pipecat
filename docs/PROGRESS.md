# PROGRESS — Qwen3-TTS megakernel + Pipecat

Single source of truth for project status. Updated 2026-05-30.

ALL PHASES COMPLETE. Validated end-to-end on RTX 5090 (Vast.ai). Demo recorded.
Submission email + README written. Branch pushed, PR #1 updated.

GPU run summary (2026-05-30):
- CUDA 12.8 toolkit installed alongside box default nvcc 13 (cu128 torch needs CUDA 12 nvcc)
- Kernel JIT-compiled on sm_120a: PASS
- Talker weights loaded + kernel trunk running: PASS
- End-to-end voice loop (mic→Deepgram→Groq→megakernel TTS→speaker): PASS, demo recorded
- Loom demo: https://www.loom.com/share/825175f04efd4a379eee975b0824fb57
- Perf: TTFC ~3.5–5.8s, RTF ~1.4–2.1 (bottleneck: code predictor PyTorch + dummy LM head)
- Streaming frame-by-frame: PASS (project constraint met)

Bugs found+fixed on GPU (now in repo):
- setup.sh: PIP="pip install" → PIP="pip" (both branches)
- kernel_backend.py: return hidden_states=(hidden,) — generate() postproc reads hidden_states[-1]
- stream.py: deadlock fix — _run() must put _SENTINEL in finally block after generate() returns
- service.py: run_tts(text, context_id) — pipecat 1.3.0 passes context_id as 2nd positional arg
- pipeline/main.py: pipecat 1.3.0 context API (LLMContext+LLMContextAggregatorPair not OpenAILLMContext)

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

### Scope decision — Option A is the deliverable
Interview submission = **CLI text→speech demo** (no mic/STT/LLM). `scripts/say.py` is the
deliverable: deterministic, no API keys, proves the megakernel. Option B (full voice loop)
is written but optional.

### Phase 5 — Kernel swap + server  🔜 CODE COMPLETE, NEEDS RTX 5090 to RUN
- `src/tts/kernel_backend.py` — `KernelTalkerTrunk`: drives the megakernel `Decoder`,
  injects `inputs_embeds` via `dec._embed_weight=vec` + `step(0)`, reads `dec._norm_out`
  as hidden, dummy [151936,1024] LM-head buffer to avoid OOB, resets on T>1 prefill, bs=1.
  `install(model, weights)` swaps the trunk. No kernel-source edits.
- `scripts/verify_kernel_trunk.py` — FIRST GPU step: replay `golden_trunk.npz` through the
  kernel, compare hidden. **Use bf16-realistic tolerance (~1e-2 / mean<0.05), not 1e-3.**
- `src/server/main.py` — full FastAPI server. Loads model, optional kernel trunk
  (`TTS_USE_KERNEL=0` -> stock HF for free A/B speed compare), streams 24 kHz int16 PCM
  frame-by-frame. Endpoints: `WS /tts`, `POST /synthesize`, `GET /tts.wav?text=`,
  `GET /health`. Batch-1 serialized (asyncio lock). Config via env (`TTS_*`).
  Solid (only fastapi/websockets deps). py_compile OK.
- `scripts/say.py` — Option-A client: WS in, streams PCM, writes `out.wav`, `--play`
  optional, prints **TTFC + RTF**. Deps: `websockets` only. py_compile OK.
- `scripts/setup.sh` — DONE. Install ORDER matters: nightly cu128 torch+torchaudio FIRST,
  then pinned deps (covering qwen-tts's deps), then `--no-deps` qwen-tts — else qwen-tts's
  unpinned torchaudio clobbers nightly torch and loses sm_120. apt sox/libsndfile. Runs
  `preflight.py` before the kernel compile. `INSTALL_PIPECAT=1` opt-in for Option B.
- `scripts/preflight.py` — fails in seconds if the env is wrong (cu128 torch, sm_120 GPU,
  torchaudio matches torch, transformers==4.57.3, hub<1.0, qwen_tts + qwen_megakernel import)
  BEFORE the download/compile burn GPU hours. `--fast` skips the slow kernel import.
- `docs/DEMO.md` — client-showcase script (cold run, A/B, restart, save-artifacts-first).
- GPU image MUST ship CUDA 12.8 toolkit (nvcc), not driver-only, or the kernel won't compile.
- Ref clip: voice-clone model needs ref_audio + transcript. **Default = Qwen's hosted demo
  clip** (clone_2.wav + its transcript, already in `colab_validate.py`); no user clip needed
  unless a specific voice is wanted.

### Phase 6 — Pipecat TTS service  🔜 WRITTEN, UNVERIFIED (pipecat API churn)
- `src/tts/service.py` — `QwenMegakernelTTSService(TTSService)`: consumes `WS /tts`,
  re-emits `TTSAudioRawFrame` framed by Started/Stopped. Version-tolerant imports.
- `src/pipeline/main.py` — Option B: mic→STT(Deepgram)→LLM(OpenAI)→TTS→speaker, import-
  guarded. Both py_compile OK but NOT run against an installed pipecat — may need a one-spot
  import fix on the box (pipecat API drifts). Needs `INSTALL_PIPECAT=1` + provider keys.

### Phases 7-8 — full pipeline / benchmarks  ⬜ NOT STARTED
Benchmark targets: TTFC<60ms (Qwen's own first-packet = 97ms, stretch), RTF<0.15,
~1000 tok/s. `say.py` already prints TTFC/RTF for the Option-A path.

---

## Git
Branch `phase1-4-talker-adaptation`, PR #1 open. Phases 1-4 + Phase-5 scaffolding
committed (5161efa). Server/client/service/pipeline + setup.sh added in a follow-up commit.
Local only (gitignored): `reference/` (vendored Qwen source), `models/` (2.51 GB weights),
`golden_trunk.npz` (in ~/Downloads).

## Immediate next actions (all on the rented RTX 5090)
1. `bash scripts/setup.sh` ; `bash scripts/download_model.sh` ; copy `golden_trunk.npz` + a
   ref clip to the box (or use the default demo clip).
2. `verify_kernel_trunk.py` against `golden_trunk.npz` — the PASS gate (bf16 tol ~1e-2).
3. Start `src/server/main.py`, run `scripts/say.py "..."` -> `out.wav` + TTFC/RTF. Demo done.
4. (optional) A/B: `TTS_USE_KERNEL=0` vs `1` to show the kernel speedup.
