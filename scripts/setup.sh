#!/bin/bash
# Provision an RTX 5090 (sm_120) box for Qwen3-TTS + qwen_megakernel.
# Run after SSH into the Vast.ai instance, from the repo root:
#   bash scripts/setup.sh
#
# Needs: NVIDIA driver + CUDA 12.8+ toolkit (nvcc) preinstalled on the image.
# The megakernel JIT-compiles on first import (-arch=sm_120a), so nvcc must be present.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# ---- sanity: GPU + CUDA toolkit ----
echo "== GPU / CUDA =="
nvidia-smi || { echo "no nvidia-smi — wrong instance"; exit 1; }
if ! command -v nvcc >/dev/null 2>&1; then
  echo "WARNING: nvcc not found. CUDA 12.8+ toolkit required to compile the megakernel."
  echo "Install the CUDA toolkit (not just the driver) before continuing."
fi
nvcc --version 2>/dev/null | tail -1 || true

# ---- python package manager (prefer uv) ----
if command -v uv >/dev/null 2>&1; then PIP="uv pip"; else PIP="pip install"; fi
echo "== using: $PIP =="

# ---- PyTorch nightly for CUDA 12.8 (Blackwell sm_120) ----
echo "== PyTorch (cu128 nightly) =="
$PIP install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# ---- pinned stack (see docs/PROGRESS.md: newer transformers breaks qwen_tts) ----
echo "== pinned deps =="
$PIP install "transformers==4.57.3" "huggingface_hub<1.0" \
             ninja accelerate safetensors soundfile librosa \
             fastapi "uvicorn[standard]" numpy

# ---- Qwen3-TTS inference package (Qwen3TTSModel, tokenizer, generate_* helpers) ----
echo "== qwen-tts =="
$PIP install "git+https://github.com/QwenLM/Qwen3-TTS.git"

# optional: faster prefill (HF parts run eager without it)
$PIP install flash-attn --no-build-isolation || echo "(flash-attn skipped; eager attn is fine)"

# ---- megakernel submodule + its requirements ----
echo "== megakernel submodule =="
git submodule update --init --recursive
[ -f kernel/requirements.txt ] && $PIP install -r kernel/requirements.txt || true

# ---- compile the megakernel now (JIT, ~minutes; cached after) ----
echo "== compiling megakernel (sm_120a) =="
python3 -c "import qwen_megakernel; print('megakernel built OK')" \
  || { echo "kernel build FAILED — check nvcc / CUDA 12.8 / sm_120a support"; exit 1; }

cat <<'EOF'

Setup done. Next:
  bash scripts/download_model.sh                  # pull Qwen3-TTS-12Hz-0.6B-Base (hf login)
  # copy golden_trunk.npz from Colab to here, then the Phase-5 gate:
  python3 scripts/verify_kernel_trunk.py --model models/Qwen3-TTS-12Hz-0.6B-Base --golden golden_trunk.npz
See docs/phase5-runbook.md.
EOF
