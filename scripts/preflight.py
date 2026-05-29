"""
Preflight env check for the RTX 5090 box — fails in seconds if anything is wrong, so we
DON'T burn GPU time discovering a broken env after a 2.5 GB download or a kernel compile.

Run after scripts/setup.sh (it calls this automatically), or any time to re-validate:
    python3 scripts/preflight.py

Checks the known-fragile env invariants (see docs/PROGRESS.md):
  - torch is the cu128 build, CUDA available, GPU is sm_120 (Blackwell / RTX 5090)
  - torchaudio is present and built against the SAME torch (no stable-wheel clobber)
  - transformers == 4.57.3   (newer breaks qwen_tts check_model_inputs)
  - huggingface_hub < 1.0
  - qwen_tts imports and exposes Qwen3TTSModel
  - qwen_megakernel imports (JIT compile happens here on first run)

Exit 0 = all green; exit 1 = at least one hard failure (printed with a fix hint).
"""

from __future__ import annotations

import sys

FAILS: list[str] = []
WARNS: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def fail(msg: str, fix: str) -> None:
    FAILS.append(msg)
    print(f"  [FAIL] {msg}\n         fix: {fix}")


def warn(msg: str) -> None:
    WARNS.append(msg)
    print(f"  [warn] {msg}")


def check_torch() -> None:
    try:
        import torch
    except Exception as e:
        fail(f"torch import failed: {e}",
             "pip install --pre torch torchaudio --index-url "
             "https://download.pytorch.org/whl/nightly/cu128")
        return

    ver = torch.__version__
    if "cu128" not in ver:
        warn(f"torch {ver} — expected a cu128 nightly build (string lacks 'cu128')")
    else:
        ok(f"torch {ver}")

    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() == False",
             "wrong instance / driver mismatch — need an RTX 5090 box with CUDA 12.8+")
        return

    try:
        major, minor = torch.cuda.get_device_capability(0)
        cap = major * 10 + minor
        name = torch.cuda.get_device_name(0)
        if cap == 120:
            ok(f"GPU {name} sm_{cap} (Blackwell)")
        else:
            warn(f"GPU {name} sm_{cap} — kernel targets sm_120 (RTX 5090); other arch may fail")
    except Exception as e:
        warn(f"could not read device capability: {e}")


def check_torchaudio() -> None:
    try:
        import torch
        import torchaudio
    except Exception as e:
        fail(f"torchaudio import failed: {e}",
             "pip install --pre torchaudio --index-url "
             "https://download.pytorch.org/whl/nightly/cu128  (must match torch)")
        return
    # stable torchaudio dragging in a stable torch is the classic clobber
    if "cu128" not in torchaudio.__version__ and "cu128" in torch.__version__:
        warn(f"torchaudio {torchaudio.__version__} not a cu128 build while torch is — "
             "possible stable-wheel clobber; reinstall torch+torchaudio from the nightly index")
    else:
        ok(f"torchaudio {torchaudio.__version__}")


def check_pin(mod: str, want: str | None, less_than: str | None = None) -> None:
    try:
        m = __import__(mod)
    except Exception as e:
        fail(f"{mod} import failed: {e}", f"pip install '{mod}'")
        return
    ver = getattr(m, "__version__", "?")
    if want is not None and ver != want:
        fail(f"{mod} == {ver}, need {want}",
             f"pip install '{mod}=={want}'  (newer breaks qwen_tts)")
    elif less_than is not None:
        try:
            from packaging.version import Version
            if Version(ver) >= Version(less_than):
                fail(f"{mod} == {ver}, need < {less_than}",
                     f"pip install '{mod}<{less_than}'")
                return
        except Exception:
            pass
        ok(f"{mod} {ver}")
    else:
        ok(f"{mod} {ver}")


def check_qwen_tts() -> None:
    try:
        from qwen_tts import Qwen3TTSModel  # noqa: F401
        ok("qwen_tts.Qwen3TTSModel importable")
    except Exception as e:
        fail(f"qwen_tts import failed: {e}",
             "pip install --no-deps git+https://github.com/QwenLM/Qwen3-TTS.git")


def check_megakernel() -> None:
    # First import JIT-compiles (slow). Skip with --fast to keep preflight quick.
    if "--fast" in sys.argv:
        warn("skipping qwen_megakernel import (--fast)")
        return
    try:
        import qwen_megakernel  # noqa: F401
        ok("qwen_megakernel importable (kernel compiled)")
    except Exception as e:
        fail(f"qwen_megakernel import failed: {e}",
             "needs nvcc + CUDA 12.8 + sm_120a; run from repo root after "
             "`git submodule update --init --recursive`")


def main() -> int:
    print("preflight: env invariants for Qwen3-TTS megakernel")
    check_torch()
    check_torchaudio()
    check_pin("transformers", want="4.57.3")
    check_pin("huggingface_hub", want=None, less_than="1.0")
    check_qwen_tts()
    check_megakernel()

    print()
    if FAILS:
        print(f"PREFLIGHT FAILED — {len(FAILS)} hard issue(s), {len(WARNS)} warning(s).")
        return 1
    print(f"PREFLIGHT OK — 0 failures, {len(WARNS)} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
