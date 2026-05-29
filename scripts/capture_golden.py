"""
Phase 3 — capture golden talker I/O from the real model (run on Colab/GPU box).

Loads the real Qwen3-TTS model, synthesizes one utterance with GREEDY decoding
(deterministic), and dumps:
  - trunk_inputs / trunk_outputs : the talker-trunk inputs_embeds and last_hidden_state
    for every trunk call (the exact pairs the megakernel must reproduce in Phase 5),
  - codes  : the [T,16] codebook tokens,
  - wav/sr : the synthesized audio (sanity that the run was real).

Needs torch + the qwen_tts package (`pip install qwen-tts` or the repo) + the model.
Run, e.g.:
    python3 scripts/capture_golden.py \
        --model models/Qwen3-TTS-12Hz-0.6B-Base \
        --text "Hello from the megakernel." --speaker <name> \
        --out golden/utt0.npz

Then in Phase 5 on the 5090, feed trunk_inputs through the kernel and assert the
output hidden matches trunk_outputs within bf16 tolerance.
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model dir")
    ap.add_argument("--text", default="Hello from the megakernel.")
    ap.add_argument("--language", default="English")
    ap.add_argument("--speaker", default=None, help="builtin speaker name (or omit)")
    ap.add_argument("--out", default="golden/utt0.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import os
    import numpy as np
    import torch
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.tts.trunk import capture_trunk_io

    # qwen_tts inference wrapper (handles tokenizer/vocoder + generate_* helpers)
    from qwen_tts import Qwen3TTSModel

    print(f"loading {args.model} ...")
    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device,
                                        dtype=torch.bfloat16)

    io, remove = capture_trunk_io(tts)

    # greedy = deterministic golden. Most builds expose generate_custom_voice.
    print("synthesizing (greedy) ...")
    wavs, sr = tts.generate_custom_voice(
        text=[args.text], language=[args.language],
        speakers=[args.speaker] if args.speaker else None,
        do_sample=False, max_new_tokens=2048,
    )
    remove()

    wav = np.asarray(wavs[0]).squeeze()
    trunk_inputs = [t.numpy() for t in io.inputs]
    trunk_outputs = [t.numpy() for t in io.outputs]
    print(f"trunk calls captured: {len(io)} "
          f"(1 prefill of T tokens + per-frame single-token steps)")
    print(f"  input[0] shape  = {trunk_inputs[0].shape}")
    print(f"  output[0] shape = {trunk_outputs[0].shape}")
    print(f"  audio: {wav.shape} @ {sr} Hz")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        wav=wav, sr=sr,
        n_calls=len(io),
        # ragged -> store as object arrays
        trunk_inputs=np.array(trunk_inputs, dtype=object),
        trunk_outputs=np.array(trunk_outputs, dtype=object),
    )
    print(f"saved golden -> {args.out}")


if __name__ == "__main__":
    main()
