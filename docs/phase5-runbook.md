# Phase 5 runbook — kernel swap on RTX 5090

Goal: run the talker trunk on the megakernel, prove it matches the golden, then stream.
Order is deliberate: **verify the kernel before building the server on top of it.**

## 0. Rent + setup (Vast.ai)
- Instance: **RTX 5090** (sm_120), Ubuntu 22.04, CUDA **12.8+**. ~$0.40-0.70/hr.
- Pick a PyTorch-nightly-cu128 image if available, else install:
  ```bash
  bash scripts/setup.sh        # (fill in: torch cu128 nightly, transformers==4.57.3,
                               #  huggingface_hub<1.0, ninja, soundfile, librosa, fastapi, uvicorn)
  ```
- The kernel JIT-compiles on first import (`qwen_megakernel`) — needs nvcc + CUDA 12.8,
  `-arch=sm_120a`. First import is slow (compile); cached after.
- Download weights: `bash scripts/download_model.sh` (hf login; gated).

## 1. Verify kernel trunk == HF trunk (DO THIS FIRST)
Copy the `golden_trunk.npz` produced on Colab to the box, then:
```bash
python3 scripts/verify_kernel_trunk.py \
    --model models/Qwen3-TTS-12Hz-0.6B-Base --golden golden_trunk.npz
```
Feeds the golden talker inputs_embeds through `KernelTalkerTrunk` and compares hidden
states to the golden outputs, in order (call 0 = prefill builds KV, rest = gen steps).
- **PASS** = mean abs diff < ~0.05, no blow-up across the sequence (bf16 tolerance).
- **FAIL** = check, in order: RoPE (theta 1e6 / cos-sin layout), weight packing order
  (11/layer), embed-injection (token_id=0 + embed_weight row), KV reset (T>1 prefill).

This isolates the kernel from everything else. If it passes, the hard part is done.

## 2. End-to-end with kernel trunk
Swap the trunk into the real model and synthesize, comparing audio to the HF reference:
```python
import torch
from qwen_tts import Qwen3TTSModel
from src.tts.weights import load_talker_weights
from src.tts.kernel_backend import install
from src.tts.stream import stream_tts_pcm

tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda", dtype=torch.bfloat16)
weights, _ = load_talker_weights(MODEL, device="cuda")
backend, restore = install(tts.model, weights, device="cuda")  # talker trunk -> kernel

pcm = b"".join(stream_tts_pcm(
    tts, generate_method="generate_voice_clone",
    text="Hello from the megakernel.", language="english",
    ref_audio=REF_WAV, ref_text=REF_TEXT, do_sample=False))
# write pcm (int16, 24kHz mono) to wav; compare to the Colab A_reference.wav
```
Greedy (`do_sample=False`) so it should closely track the reference audio.

## 3. FastAPI / WebSocket server (after 1-2 pass)
`src/server/main.py`: own the GPU + model + kernel backend; endpoint runs
`astream_tts_pcm` and pushes binary PCM frames over a WebSocket. (Not yet written —
build once the kernel trunk is verified.)

## 4. Benchmark (Phase 8)
TTFC, RTF, tok/s, e2e. Tune `chunk_frames` (latency) and `left_context` (quality vs
recompute). Note Qwen's own first-packet number is 97 ms.

## Known cost/perf notes
- Dummy [151936,1024] LM-head buffer (~311 MB) wastes ~0.3 ms/step. If RTF needs it,
  add a trunk-only launch path later (avoids the wasted vocab projection).
- One kernel instance = one stream (batch 1). Multiple concurrent voices = multiple
  instances / serialization.
- flash-attn optional (HF parts run eager without it); install for faster prefill.
