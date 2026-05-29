# Qwen3-TTS Megakernel + Pipecat Voice Pipeline

Wire AlpinDale's `qwen_megakernel` (a ~1,200-line CUDA kernel that decodes Qwen3-0.6B
at ~1,000 tok/s on an RTX 5090) as the talker-decoder backend for Qwen3-TTS, streaming
real-time speech into a Pipecat voice agent pipeline.

**Demo:** https://www.loom.com/share/825175f04efd4a379eee975b0824fb57

---

## Architecture

```
                    ┌─────────────────── RTX 5090 (Vast.ai) ───────────────────────┐
                    │                                                                │
Text prompt ──────► │  Talker trunk (megakernel)  ──► codec_head (PyTorch, 3072)   │
                    │       ▲  28 layers, 1024 hidden                               │
                    │       │  theta=1e6 RoPE, bf16                                 │
                    │       │                                                        │
                    │  dual-track input (per frame):                                │
                    │    sum(16 group embeds) + text_condition                      │
                    │                                                                │
                    │  Code Predictor (PyTorch, 5L) → groups 1-15                  │
                    │  Vocoder (PyTorch, causal) → 24 kHz PCM                      │
                    │                                                                │
                    │  FastAPI WS /tts → streams int16 PCM frame-by-frame           │
                    └─────────────────────────────────────────────────────────────┘
                                          │ WebSocket (tunnel or LAN)
                                          ▼
                    Mac / client machine
                    mic → Deepgram STT → Groq LLM → QwenMegakernelTTSService
                                                     (Pipecat TTSService)
                                                             │
                                                         speaker
```

Three-stage Qwen3-TTS pipeline:
1. **Talker LM** — predicts codebook group 0, autoregressive at 12.5 Hz. **This is where the kernel runs.**
2. **Code Predictor** — given group 0, predicts groups 1-15. Pure PyTorch.
3. **Vocoder** — 16 codebook groups → 24 kHz waveform. Pure PyTorch.

---

## Kernel Adaptation — What Changed and Why

The megakernel was written for Qwen3-0.6B (text LM). The Qwen3-TTS talker decoder
has the **same transformer shape** (28 layers, hidden 1024, 16/8 GQA heads, head_dim 128,
intermediate 3072). So the CUDA kernel runs unchanged.

All adaptation is in Python:

| Change | Where | Why |
|--------|-------|-----|
| `rope_theta`: 10,000 → 1,000,000 | `src/tts/weights.py` | Talker config sets theta=1e6. Verified: kernel uses half-split `rotate_half` == HF; MRoPE collapses (T=H=W, no vision) → identical to 1D RoPE. Proven in `scripts/verify_rope.py` (exact 0.0 diff). |
| Codec embedding swap | `src/tts/weights.py` | Text model uses tied lm_head/embed. Talker uses separate `codec_embedding` [3072,1024] + `codec_head` [3072,1024]. Loader detects both by shape. |
| Inputs-embeds injection | `src/tts/kernel_backend.py` | Talker input per step = sum of 16 codebook-group embeds + text condition (not a token lookup). Injected by setting `dec._embed_weight = input_vector` and calling `step(0)`. Layer 0 reads row 0 = our vector. |
| Read `g_normalized` as hidden | `src/tts/kernel_backend.py` | The kernel's fused argmax LM head is hardcoded to vocab=151,936. Talker uses vocab=3072 + sampling. We read `dec._norm_out` (final-normed hidden) and pass it to a PyTorch `codec_head` + sampler instead. |
| Dummy LM-head buffer | `src/tts/kernel_backend.py` | The kernel's fused LM-head still launches. A dummy [151,936 x 1024] bf16 buffer (~311 MB) is passed to avoid OOB reads. The argmax output is discarded. |

**No changes to any file in `kernel/`.** All modifications are in `src/`.

---

## Performance

Measured on RTX 5090, CUDA 12.8, PyTorch nightly cu128.

| Metric | Measured | Target | Notes |
|--------|----------|--------|-------|
| Kernel trunk tok/s | ~1,000 | ~1,000 | Trunk-only, matches megakernel spec |
| TTFC (end-to-end) | 3.5–5.8 s | < 60 ms | See bottleneck below |
| RTF (end-to-end) | 1.4–2.1 | < 0.15 | See bottleneck below |
| Frame-by-frame streaming | YES | required | Never buffers full utterance |
| Audio quality | Clean | acceptable | Validated Colab + box |

### Why RTF > 1.0 (bottleneck analysis)

The kernel's 1,000 tok/s is a **trunk-only** number. But the full TTS pipeline per frame also runs:

- **Code Predictor**: 15 sequential PyTorch forwards per frame (groups 1-15), each
  a 5-layer transformer. At 20 frames/utterance = 300 PyTorch forwards. This sets the RTF floor above 1.0.
- **Dummy LM head**: 311 MB vocab projection per step (~0.3 ms wasted).
- **Per-step Python loop**: `step()` call overhead + sampling.

The clearest optimization path (not implemented):
1. Recompile kernel with `LDG_VOCAB_SIZE=3072` — eliminates 311 MB waste, makes kernel argmax = group-0 token directly.
2. Accelerate the Code Predictor (fuse onto kernel or batch).
3. CUDA graph the decode loop to eliminate Python launch overhead per step.

---

## How to Run

### Prerequisites
- RTX 5090, Ubuntu 22.04, CUDA 12.8+ **toolkit** (nvcc required — driver-only images won't compile the kernel)
- HuggingFace account with license accepted for `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

### Step 1 — Provision the box

```bash
git clone -b phase1-4-talker-adaptation https://github.com/sureliyajd/qwen-tts-pipecat.git
cd qwen-tts-pipecat
export HF_TOKEN=hf_...
bash scripts/setup.sh        # installs deps, compiles kernel, runs preflight
bash scripts/download_model.sh
```

`setup.sh` installs torch + torchaudio from the cu128 nightly index **first**, then
installs `qwen-tts --no-deps` to prevent it pulling a stable torchaudio that would
clobber the nightly build and lose sm_120 support. Runs `scripts/preflight.py`
before the kernel compile to fail fast if the env is wrong.

### Step 2 — Start the server

```bash
export TTS_MODEL_DIR=models/Qwen3-TTS-12Hz-0.6B-Base
export TTS_REF_AUDIO="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
export TTS_REF_TEXT="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"
export TTS_USE_KERNEL=1      # 0 = stock HF talker for A/B compare
python src/server/main.py
```

### Step 3 — Option A: CLI text-to-speech demo (no API keys needed)

```bash
# open SSH tunnel on client: ssh -p PORT -N -L 8000:localhost:8000 root@HOST
pip install websockets
python scripts/say.py "Hello from the megakernel." -o out.wav
# prints: TTFC, audio duration, wall time, RTF
```

### Step 4 — Option B: Full voice pipeline (STT → LLM → TTS)

```bash
pip install "pipecat-ai[deepgram,openai,local]"
export DEEPGRAM_API_KEY=...
export GROQ_API_KEY=...      # or OPENAI_API_KEY
export TTS_WS_URL=ws://127.0.0.1:8000/tts
python src/pipeline/main.py
# speak into mic; shows transcript + per-reply TTS metrics live
```

### A/B: kernel vs stock HF

```bash
TTS_USE_KERNEL=1 python src/server/main.py   # kernel path
TTS_USE_KERNEL=0 python src/server/main.py   # HF path
python scripts/say.py "The quick brown fox." -o out.wav
# compare printed RTF between runs
```

---

## Repository Structure

```
src/
  tts/
    weights.py          # introspective talker weight loader (detects keys by shape)
    kernel_backend.py   # KernelTalkerTrunk — megakernel adapter
    trunk.py            # seam: capture_trunk_io, install_trunk_backend
    stream.py           # AudioFrameStreamer + StreamingVocoder + astream_tts_pcm
    service.py          # Pipecat QwenMegakernelTTSService (pipecat 1.3.0)
  server/
    main.py             # FastAPI WS server (WS /tts, POST /synthesize, GET /tts.wav)
  pipeline/
    main.py             # full STT -> LLM -> TTS Pipecat pipeline
scripts/
  setup.sh              # one-command GPU box provision
  preflight.py          # env sanity check (cu128 torch, sm_120, version pins)
  download_model.sh     # pull Qwen3-TTS-12Hz-0.6B-Base from HuggingFace
  say.py                # CLI demo client — prints TTFC + RTF per utterance
  verify_rope.py        # RoPE equivalence proof (stdlib only, no torch/GPU)
  verify_kernel_trunk.py # kernel vs HF hidden-state comparison on golden vectors
  colab_validate.py     # Colab validation: reference path + streaming glue check
  capture_golden.py     # capture golden trunk I/O for kernel verification
docs/
  rope-analysis.md      # Phase 1: RoPE resolution + proof
  talker-decode.md      # Phase 3: dual-track mechanism (from real source code)
  streaming.md          # Phase 4: vocoder + streaming design
  DEMO.md               # step-by-step showcase guide (cold run, A/B, restart)
  phase5-runbook.md     # GPU box runbook
  PROGRESS.md           # full status, all bugs found/fixed, all phase notes
kernel/                 # git submodule — AlpinDale/qwen_megakernel (unmodified)
```

---

## Validated

- **Phase 1 (RoPE)**: `scripts/verify_rope.py` — exact 0.0 diff, pure stdlib, no GPU
- **Phase 4 (streaming)**: bookkeeping unit-tested torch-free; Colab T4 validated (fp32 max|diff|=1.6e-6, lengths match)
- **Phase 5 (kernel on hardware)**: JIT-compiled on RTX 5090 sm_120a/CUDA 12.8; end-to-end audio confirmed
- **End-to-end loop**: mic → Deepgram STT → Groq LLM → megakernel TTS → speaker; live demo recorded

---

## Stack

| Layer | Technology |
|-------|-----------|
| GPU | NVIDIA RTX 5090 (sm_120, Blackwell) |
| CUDA | 12.8 + PyTorch nightly cu128 |
| TTS model | Qwen/Qwen3-TTS-12Hz-0.6B-Base |
| Decode kernel | AlpinDale/qwen_megakernel (unmodified) |
| Inference server | FastAPI + uvicorn |
| Voice pipeline | Pipecat 1.3.0 |
| STT | Deepgram |
| LLM | Groq (llama-3.1-8b-instant, free tier) |
