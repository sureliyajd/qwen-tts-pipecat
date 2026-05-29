"""
Phase 5 step 1 (RTX 5090) — prove the megakernel trunk == HF talker trunk.

Loads golden_trunk.npz (from colab_validate.py: per-call talker inputs_embeds +
last_hidden_state, captured greedily) and the talker weights, runs the SAME inputs
through KernelTalkerTrunk, and compares the hidden states position-by-position.

Run order matters: call 0 is the multi-token prefill (resets + builds KV), calls 1..N
are single-token gen steps that depend on that KV — so we replay them in order.

    python3 scripts/verify_kernel_trunk.py \
        --model models/Qwen3-TTS-12Hz-0.6B-Base --golden golden_trunk.npz

PASS criterion: bf16-level agreement. The kernel accumulates differently than HF, so
expect small abs diffs (~1e-2); a clean match is mean abs diff well under ~0.05 and no
blow-up across the sequence. Large/growing error => a real bug (RoPE, weight packing,
injection, or KV).
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model dir")
    ap.add_argument("--golden", required=True, help="golden_trunk.npz from Colab")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol_mean", type=float, default=0.05)
    args = ap.parse_args()

    import os
    import sys
    import numpy as np
    import torch

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.tts.weights import load_talker_weights
    from src.tts.kernel_backend import KernelTalkerTrunk

    g = np.load(args.golden, allow_pickle=True)
    trunk_in = list(g["trunk_in"])
    trunk_out = list(g["trunk_out"])
    print(f"golden: {len(trunk_in)} trunk calls "
          f"(call 0 prefill T={trunk_in[0].shape[0]}, rest single-token)")

    print("loading talker weights into kernel layout (triggers CUDA JIT build)...")
    weights, _meta = load_talker_weights(args.model, device=args.device, verbose=True)
    trunk = KernelTalkerTrunk(weights, device=args.device)

    worst = 0.0
    worst_mean = 0.0
    for i, (xin, xout) in enumerate(zip(trunk_in, trunk_out)):
        ie = torch.from_numpy(np.asarray(xin)).to(args.device)[None]   # [1,T,1024]
        out = trunk(inputs_embeds=ie).last_hidden_state[0].float().cpu().numpy()
        ref = np.asarray(xout, dtype=np.float32)
        d = np.abs(out - ref)
        mx, mean = float(d.max()), float(d.mean())
        worst = max(worst, mx)
        worst_mean = max(worst_mean, mean)
        tag = "" if mean < args.tol_mean else "  <-- HIGH"
        if i < 3 or mean >= args.tol_mean or i == len(trunk_in) - 1:
            print(f"  call {i:>3} T={out.shape[0]:<3} max|d|={mx:.4f} mean|d|={mean:.4f}{tag}")

    print(f"\nworst max|d|={worst:.4f}  worst mean|d|={worst_mean:.4f}")
    if worst_mean < args.tol_mean:
        print("PASS: kernel trunk matches HF trunk (bf16 tolerance).")
    else:
        print("FAIL: divergence too large. Check RoPE theta/layout, weight packing "
              "order, embed injection, or KV cache reset.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
