# DEMO — Qwen3-TTS megakernel (Option A: text → speech)

The interview deliverable. Text in → spoken `out.wav` out, off the megakernel, with
TTFC + RTF printed. No mic, no STT/LLM, no API keys.

---

## First run (cold box) — ~1-2 hours, do once

Budget the time: kernel JIT compile + 2.5 GB model download + golden verify. Not 30 min.

```bash
# on the RTX 5090 box (CUDA 12.8 toolkit / nvcc present — see image note below)
git clone -b phase1-4-talker-adaptation https://github.com/sureliyajd/qwen-tts-pipecat.git
cd qwen-tts-pipecat

export HF_TOKEN=hf_...                  # license accepted on the model page
bash scripts/setup.sh                   # installs + preflight + compiles kernel (fails fast if env wrong)
bash scripts/download_model.sh          # pulls Qwen3-TTS-12Hz-0.6B-Base
# copy the golden vectors captured on Colab:
#   scp golden_trunk.npz user@box:~/qwen-tts-pipecat/

# PASS GATE — kernel trunk must match the golden before anything else:
python3 scripts/verify_kernel_trunk.py \
    --model models/Qwen3-TTS-12Hz-0.6B-Base --golden golden_trunk.npz
# expect: "PASS: kernel trunk matches HF trunk" (bf16 tol ~1e-2)
```

## The demo

```bash
# reference clip (default = Qwen's hosted demo clip; or your own + exact transcript)
export TTS_MODEL_DIR=models/Qwen3-TTS-12Hz-0.6B-Base
export TTS_REF_AUDIO=https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav
export TTS_REF_TEXT="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
export TTS_USE_KERNEL=1

tmux new -d -s tts 'python src/server/main.py'    # server in background
sleep 60 && curl -s localhost:8000/health         # wait for "ready": true

python scripts/say.py "Hello from the megakernel." -o out.wav
#   -> TTFC (first chunk): NN ms
#   -> audio: N.NNs | wall: N.NNs | RTF: N.NNN | bytes: ...
```

**Save the artifacts immediately** (the GPU may be gone on a later restart — see below):
```bash
scp user@box:~/qwen-tts-pipecat/out.wav .         # pull the audio down
# screen-record the say.py TTFC/RTF line
```

## The money shot — kernel vs no-kernel A/B

```bash
# kernel (default)
python scripts/say.py "The quick brown fox jumps over the lazy dog." -o kernel.wav

# stock HF talker, same everything else
tmux kill-session -t tts
TTS_USE_KERNEL=0 tmux new -d -s tts 'python src/server/main.py' ; sleep 60
python scripts/say.py "The quick brown fox jumps over the lazy dog." -o hf.wav
```
Compare the printed RTF / TTFC — that's the megakernel speedup, same audio.

## Restart showcase (after a STOP) — ~5-10 min

Vast **STOP** keeps the disk: model + compiled kernel are cached, so no re-download/compile.
But STOP does **not** reserve the physical GPU — it may be unavailable on restart (then you
need a new instance = another cold run). **DESTROY** wipes everything.

So: keep `out.wav` + the recorded numbers locally as the can't-fail backup. If the box comes
back:
```bash
cd qwen-tts-pipecat
python3 scripts/preflight.py --fast      # 5 s: confirms env survived
tmux new -d -s tts 'python src/server/main.py' ; sleep 30
python scripts/say.py "Live from the megakernel." --play   # or -o out2.wav
```

## Image note (pick before renting)
The box needs the **CUDA 12.8 toolkit (nvcc)**, not just the driver, or the kernel won't
compile. Choose a `*-devel` / PyTorch CUDA-12.8 image. `setup.sh` warns and `preflight.py`
fails in seconds if nvcc/sm_120 is missing — but it blocks the run, so get the image right.

## Stop spending
`tmux kill-session -t tts` then STOP the instance from the Vast dashboard.
