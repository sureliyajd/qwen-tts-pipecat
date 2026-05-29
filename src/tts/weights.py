"""
Phase 2 — Qwen3-TTS talker weight loader for the qwen_megakernel.

Loads the **talker** decoder of `Qwen/Qwen3-TTS-12Hz-0.6B-Base` and packs it into
the exact layout the megakernel `Decoder` expects (see kernel/qwen_megakernel/model.py):

    weights = {
        "embed_weight":      bf16 [vocab=3072, 1024],   # talker codec embedding
        "layer_weights":     [11 tensors/layer x 28],    # flat list, kernel order
        "final_norm_weight": bf16 [1024],
        "lm_head_weight":    bf16 [3072, 1024],          # talker output head (UNtied)
        "cos_table":         bf16 [max_seq, 128],        # theta = 1e6 (see docs/rope-analysis.md)
        "sin_table":         bf16 [max_seq, 128],
    }
    meta  = {"text_proj": [...], ...}                    # extras for later phases

Design: we do NOT hardcode HF state_dict key strings (the talker prefix is not
publicly documented). Instead we autodetect by structure + shape. Two entrypoints:

    python src/tts/weights.py inspect <model_dir>   # stdlib only, no torch/GPU
    python src/tts/weights.py verify  <model_dir>   # needs torch + safetensors

`inspect` reads the safetensors header (pure stdlib) and prints every tensor name,
shape and dtype, grouped — run it once weights are downloaded to confirm detection.

Per-layer kernel order (must match _pack_layer_weights in the kernel):
    input_layernorm, q_proj, k_proj, v_proj, q_norm, k_norm, o_proj,
    post_attention_layernorm, gate_proj, up_proj, down_proj
"""

import json
import os
import re
import struct
import sys

# --- talker constants (from config.json of Qwen3-TTS-12Hz-0.6B-Base) ---------
HIDDEN = 1024
NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_SIZE = NUM_Q_HEADS * HEAD_DIM   # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 1024
INTERMEDIATE = 3072
VOCAB = 3072            # 2048 codec codes + special tokens
TEXT_HIDDEN = 2048
ROPE_THETA = 1_000_000.0
MAX_SEQ_LEN = 2048      # matches kernel KV cache sizing (config allows 32768)

# Per-layer suffixes in kernel pack order.
LAYER_SUFFIXES = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


# =============================================================================
# Pure-stdlib safetensors header reader (no torch, no GPU, no network)
# =============================================================================
def read_safetensors_header(path: str) -> dict:
    """Return {name: {"dtype": str, "shape": [...]}} from a .safetensors file.

    Reads only the JSON header (first bytes), not the tensor data.
    """
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in header.items()}


def _find_model_file(model_dir: str) -> str:
    """Locate the main talker safetensors (not the speech_tokenizer one)."""
    cand = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(cand):
        return cand
    # fallback: first top-level .safetensors that is not under speech_tokenizer/
    for root, _, files in os.walk(model_dir):
        if "speech_tokenizer" in root:
            continue
        for fn in files:
            if fn.endswith(".safetensors"):
                return os.path.join(root, fn)
    raise FileNotFoundError(f"no model.safetensors under {model_dir}")


# =============================================================================
# Structure detection (works on a header dict; no tensors loaded)
# =============================================================================
def detect_talker(header: dict) -> dict:
    """Return detected key strings:
    {"prefix", "root", "embed", "final_norm", "lm_head", "text_proj"(optional)}.

    prefix = "<...>.layers" of the talker decoder (28 layers, hidden 1024).
    """
    keys = list(header.keys())

    # 1) find every "*.layers" group via the input_layernorm marker
    pat = re.compile(r"^(.*\.layers)\.(\d+)\.input_layernorm\.weight$")
    groups: dict[str, set] = {}
    for k in keys:
        m = pat.match(k)
        if m:
            groups.setdefault(m.group(1), set()).add(int(m.group(2)))

    if not groups:
        raise RuntimeError(
            "no '*.layers.N.input_layernorm.weight' keys found. "
            "Run `inspect` and check the actual naming."
        )

    # 2) pick the talker: the group with NUM_LAYERS layers whose q_proj is [2048,1024]
    prefix = None
    for g, idxs in groups.items():
        if len(idxs) != NUM_LAYERS:
            continue
        qk = f"{g}.0.self_attn.q_proj.weight"
        if qk in header and tuple(header[qk]["shape"]) == (Q_SIZE, HIDDEN):
            prefix = g
            break
    if prefix is None:
        # fallback: the group with the most layers
        prefix = max(groups, key=lambda g: len(groups[g]))
        sys.stderr.write(
            f"[warn] exact talker match failed; falling back to '{prefix}' "
            f"({len(groups[prefix])} layers). Verify with `inspect`.\n"
        )

    root = prefix[: -len(".layers")] if prefix.endswith(".layers") else prefix

    def find_by_shape(shape, name_must=(), name_not=()):
        hits = []
        for k, v in header.items():
            if tuple(v["shape"]) != tuple(shape):
                continue
            if any(s not in k for s in name_must):
                continue
            if any(s in k for s in name_not):
                continue
            hits.append(k)
        return hits

    # The trunk lives under "<root>.layers"; embed/norm/head are direct children
    # of <root>. The code predictor is a separate subtree (talker.code_predictor.*)
    # whose tensors we must NOT pick up here — exclude it explicitly.
    other_subtree = f"{root}.code_predictor"

    def under_root(k):
        return k.startswith(root + ".") and not k.startswith(other_subtree)

    # embed: [VOCAB, HIDDEN] under root with 'embed' in name
    embed = ([k for k in find_by_shape([VOCAB, HIDDEN], name_must=["embed"]) if under_root(k)]
             or [k for k in find_by_shape([VOCAB, HIDDEN], name_not=["head", "output"]) if under_root(k)])
    # codec head: [VOCAB, HIDDEN] with head/output in name (NOT under code_predictor)
    lm_head = [k for k in (find_by_shape([VOCAB, HIDDEN], name_must=["head"])
                           + find_by_shape([VOCAB, HIDDEN], name_must=["output"]))
               if not k.startswith(other_subtree)]
    # final norm: [HIDDEN] direct child of root (not per-layer, not code predictor)
    norm = [k for k in keys
            if tuple(header[k]["shape"]) == (HIDDEN,)
            and under_root(k) and ".layers." not in k and "norm" in k]

    # text path (used in PyTorch for conditioning prefill, not in the kernel):
    #   text_embedding [text_vocab, 2048]  ->  text_projection MLP (2048->2048->1024)
    text_embed = [k for k, v in header.items()
                  if "text_embed" in k and TEXT_HIDDEN in v["shape"]]
    text_proj = sorted(k for k in keys if "text_projection" in k)  # fc1/fc2 weight+bias

    def one(lst, label, required=True):
        if not lst:
            if required:
                raise RuntimeError(f"could not detect {label}; run `inspect`.")
            return None
        if len(lst) > 1:
            sys.stderr.write(f"[warn] multiple {label} candidates: {lst}; using {lst[0]}\n")
        return lst[0]

    return {
        "prefix": prefix,
        "root": root,
        "embed": one(embed, "codec_embedding"),
        "final_norm": one(norm, "final norm"),
        "lm_head": one(lm_head, "codec_head"),
        "text_embed": one(text_embed, "text_embedding", required=False),
        "text_proj": text_proj,  # list: [...linear_fc1.bias/weight, ...fc2.bias/weight]
    }


# =============================================================================
# RoPE tables (theta = 1e6, see docs/rope-analysis.md — proven by verify_rope.py)
# =============================================================================
def build_rope_tables(theta=ROPE_THETA, head_dim=HEAD_DIM, max_seq=MAX_SEQ_LEN):
    import torch
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cos = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).contiguous()
    sin = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).contiguous()
    return cos, sin


# =============================================================================
# Full loader (needs torch + safetensors; run on the GPU box)
# =============================================================================
def load_talker_weights(model_dir: str, device: str = "cuda", verbose: bool = True):
    """Load + pack talker weights into the kernel Decoder layout."""
    import torch
    from safetensors import safe_open

    path = _find_model_file(model_dir)
    header = read_safetensors_header(path)
    det = detect_talker(header)
    if verbose:
        print(f"detected talker prefix: {det['prefix']}")
        print(f"  embed      = {det['embed']}")
        print(f"  final_norm = {det['final_norm']}")
        print(f"  lm_head    = {det['lm_head']}")
        print(f"  text_embed = {det['text_embed']}")
        print(f"  text_proj  = {det['text_proj']}")

    prefix = det["prefix"]

    with safe_open(path, framework="pt", device=device) as f:
        def get(name):
            return f.get_tensor(name).to(torch.bfloat16).contiguous()

        layer_weights = []
        for i in range(NUM_LAYERS):
            for suf in LAYER_SUFFIXES:
                layer_weights.append(get(f"{prefix}.{i}.{suf}"))

        embed_weight = get(det["embed"])
        final_norm_weight = get(det["final_norm"])
        lm_head_weight = get(det["lm_head"])
        # text conditioning path (PyTorch-side, Phase 3): embed + 2-layer projection
        text_embed = get(det["text_embed"]) if det["text_embed"] else None
        text_proj = {k.split(".")[-2] + "." + k.split(".")[-1]: get(k)
                     for k in det["text_proj"]}  # e.g. "linear_fc1.weight"

    cos_table, sin_table = build_rope_tables()
    cos_table = cos_table.to(device)
    sin_table = sin_table.to(device)

    _sanity_check(embed_weight, final_norm_weight, lm_head_weight, layer_weights)

    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=final_norm_weight,
        lm_head_weight=lm_head_weight,
        cos_table=cos_table,
        sin_table=sin_table,
    )
    meta = dict(text_embed=text_embed, text_proj=text_proj, detected=det)
    return weights, meta


def _sanity_check(embed, final_norm, lm_head, layer_weights):
    assert tuple(embed.shape) == (VOCAB, HIDDEN), embed.shape
    assert tuple(final_norm.shape) == (HIDDEN,), final_norm.shape
    assert tuple(lm_head.shape) == (VOCAB, HIDDEN), lm_head.shape
    assert len(layer_weights) == NUM_LAYERS * len(LAYER_SUFFIXES)
    expect = {
        0: (HIDDEN,), 1: (Q_SIZE, HIDDEN), 2: (KV_SIZE, HIDDEN), 3: (KV_SIZE, HIDDEN),
        4: (HEAD_DIM,), 5: (HEAD_DIM,), 6: (HIDDEN, Q_SIZE), 7: (HIDDEN,),
        8: (INTERMEDIATE, HIDDEN), 9: (INTERMEDIATE, HIDDEN), 10: (HIDDEN, INTERMEDIATE),
    }
    for j, shp in expect.items():
        got = tuple(layer_weights[j].shape)
        assert got == shp, f"layer0 tensor {j} ({LAYER_SUFFIXES[j]}): {got} != {shp}"


# =============================================================================
# CLIs
# =============================================================================
def _cli_inspect(model_dir: str):
    path = _find_model_file(model_dir)
    header = read_safetensors_header(path)
    print(f"# {path}\n# {len(header)} tensors\n")
    layer0, nonlayer, otherlayers = [], [], 0
    for k in sorted(header):
        if re.search(r"\.layers\.\d+\.", k):
            if re.search(r"\.layers\.0\.", k):
                layer0.append(k)
            else:
                otherlayers += 1
        else:
            nonlayer.append(k)
    print("## non-layer tensors")
    for k in nonlayer:
        print(f"  {k:<60} {header[k]['dtype']:<6} {header[k]['shape']}")
    print("\n## layer 0 tensors (pattern repeats per layer)")
    for k in layer0:
        print(f"  {k:<60} {header[k]['dtype']:<6} {header[k]['shape']}")
    print(f"\n## (+{otherlayers} more tensors in other layers)")
    try:
        det = detect_talker(header)
        print("\n## detection result")
        for kk, vv in det.items():
            print(f"  {kk:<12} = {vv}")
    except Exception as e:
        print(f"\n[detection failed] {e}")


def _cli_verify(model_dir: str):
    weights, meta = load_talker_weights(model_dir, device="cpu")
    print("\nOK: shapes validated. Layout matches kernel Decoder.")
    print(f"  layer_weights: {len(weights['layer_weights'])} tensors "
          f"({NUM_LAYERS} layers x {len(LAYER_SUFFIXES)})")
    print(f"  cos/sin: {tuple(weights['cos_table'].shape)} theta={ROPE_THETA:g}")
    te = meta["text_embed"]
    print(f"  text_embed: {None if te is None else tuple(te.shape)}")
    print(f"  text_proj: { {k: tuple(v.shape) for k, v in meta['text_proj'].items()} }")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("inspect", "verify"):
        print("usage: python src/tts/weights.py {inspect|verify} <model_dir>")
        raise SystemExit(2)
    {"inspect": _cli_inspect, "verify": _cli_verify}[sys.argv[1]](sys.argv[2])
