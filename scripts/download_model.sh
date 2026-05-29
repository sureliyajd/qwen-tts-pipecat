#!/bin/bash
# Download Qwen3-TTS talker weights from HuggingFace.
# Free + no GPU needed — run anywhere, then verify the loader with:
#   python3 src/tts/weights.py inspect models/Qwen3-TTS-12Hz-0.6B-Base
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
DEST="${2:-models/$(basename "$MODEL")}"

echo "Model : $MODEL"
echo "Dest  : $DEST"

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Installing huggingface_hub CLI..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install -U "huggingface_hub"
  elif command -v pip >/dev/null 2>&1; then
    pip install -U "huggingface_hub"
  else
    python3 -m pip install -U "huggingface_hub"
  fi
fi

DL="hf"; command -v hf >/dev/null 2>&1 || DL="huggingface-cli"

# NOTE: model may be gated — accept the license on the HF page and `hf auth login` first.
"$DL" download "$MODEL" \
  --local-dir "$DEST" \
  --exclude "*.bin" \
  --exclude "original/*"

echo
echo "Done. Inspect detected keys (stdlib only, no torch):"
echo "  python3 src/tts/weights.py inspect $DEST"
echo "Validate shapes against the kernel layout (needs torch+safetensors):"
echo "  python3 src/tts/weights.py verify  $DEST"
