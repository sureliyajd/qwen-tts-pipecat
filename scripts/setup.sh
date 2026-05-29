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

# ---- system libs (sox binary backs qwen-tts's `sox` python dep) ----
if command -v apt-get >/dev/null 2>&1; then
  echo "== apt: sox + libsndfile =="
  (apt-get update -qq && apt-get install -y -qq sox libsox-dev libsndfile1) \
    || echo "(apt failed — install sox manually if torchaudio/sox errors appear)"
fi

# ---- PyTorch nightly for CUDA 12.8 (Blackwell sm_120) ----
# torch AND torchaudio MUST come from the same nightly cu128 index, BEFORE qwen-tts.
# qwen-tts depends on unpinned torchaudio; if pip resolves it later it pulls a STABLE
# torchaudio that drags in a stable torch and clobbers the nightly cu128 build (no sm_120).
echo "== PyTorch + torchaudio (cu128 nightly) =="
$PIP install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# ---- pinned stack (see docs/PROGRESS.md: newer transformers breaks qwen_tts) ----
# These cover qwen-tts's own deps too (transformers==4.57.3, accelerate==1.12.0, gradio,
# librosa, soundfile, sox, onnxruntime, einops) so the --no-deps install below is safe.
echo "== pinned deps =="
$PIP install "transformers==4.57.3" "huggingface_hub<1.0" "accelerate==1.12.0" \
             ninja safetensors soundfile librosa sox onnxruntime einops gradio \
             fastapi "uvicorn[standard]" numpy websockets

# ---- Qwen3-TTS inference package (Qwen3TTSModel, tokenizer, generate_* helpers) ----
# --no-deps: deps already installed above; this prevents qwen-tts from pulling a stable
# torch/torchaudio or bumping transformers/huggingface_hub off their pins.
echo "== qwen-tts (--no-deps) =="
$PIP install --no-deps "git+https://github.com/QwenLM/Qwen3-TTS.git"

# re-assert the hub pin in case anything above nudged it (cheap; PROGRESS: needs <1.0)
$PIP install "huggingface_hub<1.0"

# optional: faster prefill (HF parts run eager without it)
$PIP install flash-attn --no-build-isolation || echo "(flash-attn skipped; eager attn is fine)"

# optional: full voice pipeline (src/pipeline/main.py). Not needed for the server or the
# scripts/say.py demo. Skipped by default; set INSTALL_PIPECAT=1 to include.
if [ "${INSTALL_PIPECAT:-0}" = "1" ]; then
  echo "== pipecat (full pipeline) =="
  $PIP install "pipecat-ai[deepgram,openai,local,silero]" || echo "(pipecat install failed)"
fi

# ---- megakernel submodule + its requirements ----
echo "== megakernel submodule =="
git submodule update --init --recursive
[ -f kernel/requirements.txt ] && $PIP install -r kernel/requirements.txt || true

# ---- preflight: fail loud NOW if the env is wrong (before download/compile) ----
echo "== preflight =="
python3 scripts/preflight.py || { echo "preflight FAILED — fix the env before continuing"; exit 1; }

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
