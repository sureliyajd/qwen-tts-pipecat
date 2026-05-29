"""
Phase 1 proof: the megakernel's RoPE is correct for the Qwen3-TTS talker,
given only a change of rope_theta (10000 -> 1000000) when building cos/sin.

Pure Python stdlib. No torch, no numpy, no GPU, no weights, no network. Run:

    python3 scripts/verify_rope.py

What it checks
--------------
1. apply_interleaved_mrope (Qwen3-VL / Omni TMRoPE) is an IDENTITY op when the
   three position axes (T, H, W) carry the same scalar position -- which is the
   talker's case, since it has no image/video input.
2. The kernel's half-split rotate is equivalent to HF's rotate_half.
3. End-to-end: kernel-style RoPE == HF talker RoPE for random q/k over many
   positions. Asserts max abs diff < 1e-9.

If this passes, Phase 1 is done: no CUDA changes, only the theta constant.
"""

import math
import random

HEAD_DIM = 128
HALF = HEAD_DIM // 2  # 64
THETA = 1_000_000.0  # talker rope_theta (text model uses 10_000)
MROPE_SECTION = [24, 20, 20]  # sums to HALF = 64


def inv_freq(theta=THETA, dim=HEAD_DIM):
    return [1.0 / (theta ** ((2 * i) / dim)) for i in range(dim // 2)]


# --- HF reference ----------------------------------------------------------
def rotate_half(x):
    # x: list[dim]; returns [-x[half:], x[:half]]
    return [-v for v in x[HALF:]] + x[:HALF]


def apply_interleaved_mrope(freqs3, mrope_section):
    """freqs3: [3][half]. Reassemble T/H/W per Qwen3-VL. Returns [half]."""
    freqs_t = list(freqs3[0])  # temporal everywhere by default
    for dim, offset in ((1, 1), (2, 2)):  # H, W
        length = mrope_section[dim] * 3
        for idx in range(offset, length, 3):
            freqs_t[idx] = freqs3[dim][idx]
    return freqs_t


def hf_rope(q, pos3):
    """pos3: [3] scalar positions (T,H,W). q: [dim]. Returns rotated q [dim]."""
    ifr = inv_freq()
    freqs3 = [[pos3[a] * f for f in ifr] for a in range(3)]  # [3][half]
    freqs_t = apply_interleaved_mrope(freqs3, MROPE_SECTION)  # [half]
    emb = freqs_t + freqs_t  # [dim]: cat(freqs, freqs)
    cos = [math.cos(e) for e in emb]
    sin = [math.sin(e) for e in emb]
    rh = rotate_half(q)
    return [q[i] * cos[i] + rh[i] * sin[i] for i in range(HEAD_DIM)]


# --- Kernel-style RoPE (kernel.cu + model.py) ------------------------------
def kernel_rope(q, pos):
    """pos: scalar position. Mirrors model.py cos/sin + kernel apply."""
    ifr = inv_freq()
    freqs = [pos * f for f in ifr]  # [half]
    cos = [math.cos(x) for x in freqs] + [math.cos(x) for x in freqs]  # repeat(1,2)
    sin = [math.sin(x) for x in freqs] + [math.sin(x) for x in freqs]
    out = [0.0] * HEAD_DIM
    for i in range(HALF):  # q[i]*cos[i] - q[i+half]*sin[i]
        out[i] = q[i] * cos[i] - q[i + HALF] * sin[i]
    for i in range(HALF, HEAD_DIM):  # q[i]*cos[i] + q[i-half]*sin[i]
        out[i] = q[i] * cos[i] + q[i - HALF] * sin[i]
    return out


def main():
    random.seed(0)

    # 1) MRoPE identity when all 3 axes share the same position
    pos = 7
    ifr = inv_freq()
    freqs3 = [[pos * f for f in ifr] for _ in range(3)]  # equal axes
    freqs_t = apply_interleaved_mrope(freqs3, MROPE_SECTION)
    mrope_noop = max(abs(freqs_t[i] - freqs3[0][i]) for i in range(HALF))
    print(f"[1] MRoPE reassembly delta (equal axes): {mrope_noop:.2e}  -> identity")

    # 2 & 3) Full RoPE over many positions, random q/k
    max_diff = 0.0
    for pos in range(0, 2048, 17):
        q = [random.gauss(0, 1) for _ in range(HEAD_DIM)]
        k = [random.gauss(0, 1) for _ in range(HEAD_DIM)]
        for vec in (q, k):
            a = hf_rope(vec, [pos, pos, pos])
            b = kernel_rope(vec, pos)
            d = max(abs(a[i] - b[i]) for i in range(HEAD_DIM))
            max_diff = max(max_diff, d)
    print(f"[2/3] max|HF_rope - kernel_rope| over positions = {max_diff:.2e}")

    tol = 1e-9
    assert mrope_noop < tol, "MRoPE NOT identity -> talker has non-scalar positions"
    assert max_diff < tol, "kernel RoPE != HF RoPE"
    print(f"\nPASS (tol={tol}). Kernel RoPE is correct for the talker.")
    print("Only change needed: build cos/sin with theta=1_000_000 in model.py.")


if __name__ == "__main__":
    main()
