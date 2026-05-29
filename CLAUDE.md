# Project: Qwen3-TTS Megakernel + Pipecat Voice Pipeline

## Overview

Wire AlpinDale's `qwen_megakernel` (CUDA decode kernel for Qwen3 on RTX 5090)
as the inference backend for Qwen3-TTS inside a Pipecat real-time voice pipeline.

---

## Target Hardware

| Item             | Spec                          |
|------------------|-------------------------------|
| GPU              | NVIDIA RTX 5090               |
| CUDA Arch        | sm_120 (Blackwell)            |
| CUDA Toolkit     | 12.8+                         |
| OS               | Ubuntu 22.04 (Vast.ai)        |

---

## Stack

| Layer             | Technology                         |
|-------------------|------------------------------------|
| Language          | Python 3.11                        |
| Deep Learning     | PyTorch nightly (cu128)            |
| Voice Pipeline    | Pipecat                            |
| Inference Server  | FastAPI + uvicorn                  |
| TTS Model         | Qwen/Qwen3-TTS (HuggingFace)       |
| Decode Backend    | qwen_megakernel (./kernel/)        |

---

## Project Structure

```
qwen-tts-pipecat/
├── CLAUDE.md                   # you are here — read this first always
├── .cursorrules                # Cursor AI context
├── kernel/                     # git submodule — AlpinDale's megakernel
│                               # DO NOT modify core kernel files
├── src/
│   ├── server/
│   │   └── main.py             # FastAPI streaming inference server
│   ├── tts/
│   │   └── service.py          # Pipecat-compatible TTS service wrapper
│   └── pipeline/
│       └── main.py             # Full STT → LLM → TTS Pipecat pipeline
├── benchmarks/
│   └── measure.py              # Measures TTFC, RTF, tok/s, e2e latency
├── scripts/
│   ├── setup.sh                # Install deps + compile kernel on server
│   └── download_model.sh       # Pull Qwen3-TTS weights from HuggingFace
└── docs/                       # Architecture notes and decisions
```

---

## Performance Targets

| Metric           | Target     | Description                                      |
|------------------|------------|--------------------------------------------------|
| tok/s            | ~1000      | Decode speed from megakernel                     |
| TTFC             | < 60ms     | Time to first audio chunk                        |
| RTF              | < 0.15     | 1s of audio generated in < 150ms                |
| End-to-end       | < 300ms    | Speak → hear response                            |

---

## Key Constraints

- Audio MUST stream frame-by-frame to Pipecat — NEVER buffer the full utterance
- Megakernel is **bfloat16 only** — do not add quantization
- Only adapt the **talker decoder** stage of Qwen3-TTS (not the codebook generator)
- All CUDA code must target **sm_120** (Blackwell — RTX 5090 only)
- Use **async/await** for all Python streaming code

---

## Megakernel Context (kernel/)

- Architecture: 128 persistent thread blocks × 512 threads
- Single non-cooperative kernel launch
- Model: Qwen3-0.6B in bfloat16
- Output: single-token argmax per step (autoregressive decode on host)
- Performance: ~1000 tok/s, 0.97ms/step, 71% of theoretical GDDR7 bandwidth

Before modifying anything in kernel/, fully read and understand:
1. The main CUDA kernel file
2. The kernel README
3. How weights are loaded and passed in
4. Input/output tensor shapes and dtypes

---

## How to Run (on Vast.ai server)

```bash
# Step 1 — Install deps and compile the CUDA kernel
bash scripts/setup.sh

# Step 2 — Download Qwen3-TTS model weights
bash scripts/download_model.sh

# Step 3 — Start the inference server
python src/server/main.py

# Step 4 — Start the full voice pipeline
python src/pipeline/main.py
```

---

## Implementation Order

1. Understand kernel input/output interface
2. Adapt kernel for Qwen3-TTS talker decoder weights
3. Build FastAPI streaming inference server (src/server/)
4. Build Pipecat TTS service wrapper (src/tts/)
5. Wire full pipeline STT → LLM → TTS (src/pipeline/)
6. Write benchmark scripts (benchmarks/)
7. Measure and document real numbers

---

## Current Status

See `docs/PROGRESS.md` for the full, authoritative status.

- [x] Kernel submodule added and understood
- [x] RoPE resolved — kernel needs no CUDA change, theta=1e6 only (`docs/rope-analysis.md`)
- [x] Talker weight loader — introspective, confirmed vs real weights (`src/tts/weights.py`)
- [x] Decode/dual-track mechanism resolved (`docs/talker-decode.md`); trunk seam (`src/tts/trunk.py`)
- [x] Code predictor + vocoder + streaming glue (`src/tts/stream.py`)
- [x] Colab validation PASS — reference audio correct, streaming exact (fp32 diff 1.6e-6)
- [~] Qwen3-TTS talker decoder adapted — kernel backend written, awaiting RTX 5090 verify
- [ ] Inference server built (FastAPI/WS — after kernel verified)
- [ ] Pipecat TTS service wired (`astream_tts_pcm` is the hook)
- [ ] End-to-end pipeline working
- [ ] Benchmarks measured and documented
- [ ] README written with architecture decisions

Real model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (NOT `Qwen/Qwen3-TTS`). Base = voice-clone only.
Env pins: transformers==4.57.3, huggingface_hub<1.0.

---

## References

| Resource              | URL                                                        |
|-----------------------|------------------------------------------------------------|
| Megakernel source     | github.com/AlpinDale/qwen_megakernel                       |
| Megakernel blog post  | blog.alpindale.net/posts/5090_decode_optimization/         |
| Qwen3-TTS model       | huggingface.co/Qwen/Qwen3-TTS                              |
| Pipecat docs          | docs.pipecat.ai                                            |