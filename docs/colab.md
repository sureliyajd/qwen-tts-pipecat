# Colab validation (free, no RTX 5090)

Goal: prove the reference path (text -> audio) works and our streaming glue matches it,
BEFORE renting the GPU. Any Colab GPU (T4) is fine — the megakernel is NOT used here.

Runtime > Change runtime type > GPU.

### Cell 1 — install
```python
!pip install -q "git+https://github.com/QwenLM/Qwen3-TTS.git"
!pip install -q -U soundfile librosa
!pip install -q "transformers==4.57.3" "huggingface_hub<1.0"  # exact pins — qwen_tts pyproject; transformers 4.57.3 needs hub<1.0; newer transformers breaks check_model_inputs()
```
Do NOT `-U` transformers — a newer version raises
`TypeError: check_model_inputs() missing 1 required positional argument: 'func'`.
If you already upgraded it this session, reinstall the pin above then Runtime > Restart session.

### Cell 2 — HF login (model is gated)
```python
from huggingface_hub import login
login()  # paste your token (the same one that worked locally)
```

### Cell 3 — download the model
```python
from huggingface_hub import snapshot_download
MODEL = snapshot_download("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
print(MODEL)
```

### Cell 4 — upload the validator
Upload `scripts/colab_validate.py` (left panel > Files > upload), or:
```python
from google.colab import files; files.upload()   # pick colab_validate.py
```

### Cell 5 — check capabilities
```python
import torch
from qwen_tts import Qwen3TTSModel
tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda", dtype=torch.bfloat16)
print("speakers:", tts.get_supported_speakers())    # [] for *-Base -> clone only
print("languages:", tts.get_supported_languages())  # lowercase, e.g. 'english'
```
The `*-Base` model returns `speakers: []` — it has no named voices. It synthesizes by
cloning a reference clip, so Cell 6 passes `--ref_audio`/`--ref_text` (defaults provided),
not `--speaker`. Languages are lowercase.

### Cell 6 — run the 3 checks
```python
!python colab_validate.py --model "$MODEL" \
    --text "Hello from the megakernel." \
    --language english --out_dir out
```
(Validator defaults `--ref_audio`/`--ref_text` to Qwen's hosted demo clip. Override with
your own `--ref_audio URL_or_path --ref_text "transcript"` if you want a different voice.)

### Cell 7 — listen
```python
import IPython.display as ipd
print("reference:"); ipd.display(ipd.Audio("out/A_reference.wav"))
print("streamed:");  ipd.display(ipd.Audio("out/C_stream.wav"))
```

## What to look for
- **[A]** prints audio shape @ 24000 Hz and `A_reference.wav` plays correct speech ->
  the reference pipeline works.
- **[B]** `golden_trunk.npz` saved — `trunk_in`/`trunk_out` are the exact talker-trunk
  inputs_embeds and last_hidden_state per call. These are the golden pairs the megakernel
  must reproduce in Phase 5. Download and keep it.
- **[C]** lengths must be equal (proves chunk alignment), and `max|diff|` between full
  decode and our streamed decode should be `< 1e-3` in the default fp32 check. The vocoder
  is fully causal, so chunked-with-left-context decode equals full decode in exact
  arithmetic — any residual is numerical. In bf16 (`--no_fp32_check`) expect ~1e-2; that
  is rounding, not a bug. A length mismatch is a real glue bug. `--left_context` only
  matters for clips longer than the context window (short clips already use full history).

Download `out/golden_trunk.npz` (and the wavs) before the runtime expires — Phase 5 uses
the golden pairs to verify the kernel trunk numerically.
```python
from google.colab import files; files.download("out/golden_trunk.npz")
```
