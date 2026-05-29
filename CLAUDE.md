# CLAUDE.md

## What This Project Does
Wire AlpinDale's qwen_megakernel (CUDA decode kernel for Qwen3 on RTX 5090)
as the inference backend for Qwen3-TTS inside a Pipecat voice pipeline.

## Target Hardware
- GPU: NVIDIA RTX 5090
- CUDA Architecture: sm_120 (Blackwell)
- CUDA Toolkit: 12.8+
- OS: Ubuntu 22.04 on Vast.ai

## Stack
- Python 3.11
- PyTorch nightly (cu128)
- Pipecat (voice pipeline framework)
- FastAPI + uvicorn (inference server)
- Qwen3-TTS (HuggingFace: Qwen/Qwen3-TTS)
- qwen_megakernel (in ./kernel/) — CUDA megakernel for fast decode

## Project Structure
```
qwen-tts-pipecat/
├── CLAUDE.md              # you are here
├── .cursorrules           # Cursor AI context
├── kernel/                # git submodule — AlpinDale's megakernel (DO NOT MODIFY core kernel)
├── src/
│   ├── server/            # FastAPI streaming inference server
│   ├── tts/               # Pipecat-compatible TTS service wrapper
│   └── pipeline/          # Full STT → LLM → TTS Pipecat pipeline
├── benchmarks/            # Scripts to measure TTFC, RTF, tok/s
├── scripts/               # setup.sh, download_model.sh etc
└── docs/                  # Architecture notes
```

## Key Constraints
- Audio must stream frame-by-frame to Pipecat — NEVER buffer full utterance
- Target TTFC < 60ms, RTF < 0.15
- Megakernel is bfloat16 only — no quantization
- Only adapt the talker decoder stage of Qwen3-TTS (not the codebook generator)

## Performance Targets
| Metric | Target |
|--------|--------|
| TTFC   | < 60ms |
| RTF    | < 0.15 |
| tok/s  | ~1000  |

## How to Run (once on Vast.ai server)
1. `bash scripts/setup.sh`          — install deps, compile kernel
2. `bash scripts/download_model.sh` — pull Qwen3-TTS weights
3. `python src/server/main.py`      — start inference server
4. `python src/pipeline/main.py`    — start full voice pipeline

## Current Status
- [x] Kernel submodule added
- [ ] Inference server built
- [ ] Qwen3-TTS talker decoder adapted
- [ ] Pipecat pipeline wired
