"""
FastAPI streaming inference server for Qwen3-TTS on the qwen_megakernel.

Owns the GPU, the model and (optionally) the kernel-backed talker trunk. Exposes two
streaming endpoints that emit 24 kHz mono int16 PCM **frame by frame** as the talker
generates — never buffering the whole utterance (project constraint):

  - WS   /tts            : client sends one JSON request, server streams binary PCM
                           frames, then a final text frame {"event":"done"}.
  - POST /synthesize     : same audio as a chunked HTTP StreamingResponse (raw PCM).
  - GET  /tts.wav?text=  : convenience — streams a WAV (header + chunked PCM) so you can
                           `curl ... -o out.wav` or open it in a browser. Still streamed.
  - GET  /health         : model/config status.

The model is `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (voice-clone only), so every request needs
a reference audio + its transcript. Defaults come from env; a request may override them.

Run (on the RTX 5090 box, AFTER the kernel is verified — see docs/phase5-runbook.md):

    export TTS_MODEL_DIR=models/Qwen3-TTS-12Hz-0.6B-Base
    export TTS_REF_AUDIO=ref.wav
    export TTS_REF_TEXT="the transcript of ref.wav"
    export TTS_USE_KERNEL=1          # 0 = pure-HF talker (A/B compare without the kernel)
    python src/server/main.py        # or: uvicorn src.server.main:app --host 0.0.0.0 --port 8000

One kernel instance is batch-1, so concurrent requests are serialized by a lock.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
from contextlib import asynccontextmanager

# Allow `python src/server/main.py` (puts src/server on sys.path) to find the `src`
# package — same bootstrap the scripts use. Harmless under `uvicorn src.server.main`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

# stream.py is torch-free at import; safe to import here.
from src.tts.stream import SAMPLE_RATE, astream_tts_pcm

# --- config (env) ------------------------------------------------------------
MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models/Qwen3-TTS-12Hz-0.6B-Base")
REF_AUDIO = os.environ.get("TTS_REF_AUDIO", "")
REF_TEXT = os.environ.get("TTS_REF_TEXT", "")
LANGUAGE = os.environ.get("TTS_LANGUAGE", "english")
USE_KERNEL = os.environ.get("TTS_USE_KERNEL", "1") not in ("0", "false", "False", "")
DEVICE = os.environ.get("TTS_DEVICE", "cuda")
# greedy by default for reproducible demos; set TTS_DO_SAMPLE=1 for sampled voices
DO_SAMPLE = os.environ.get("TTS_DO_SAMPLE", "0") not in ("0", "false", "False", "")
CHUNK_FRAMES = int(os.environ.get("TTS_CHUNK_FRAMES", "6"))
LEFT_CONTEXT = int(os.environ.get("TTS_LEFT_CONTEXT", "25"))

# Populated at startup.
STATE: dict = {"model": None, "restore": None, "backend": None}
_GPU_LOCK = asyncio.Lock()  # batch-1 kernel -> one synthesis at a time


# --- model load --------------------------------------------------------------
def build_model():
    """Load Qwen3-TTS and, if TTS_USE_KERNEL, swap the talker trunk for the megakernel.

    Imports torch / qwen_tts / kernel lazily so the module imports on a CPU box.
    """
    import torch
    from qwen_tts import Qwen3TTSModel

    print(f"[server] loading {MODEL_DIR} on {DEVICE} (dtype=bfloat16) ...", flush=True)
    tts = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map=DEVICE, dtype=torch.bfloat16)

    restore = backend = None
    if USE_KERNEL:
        from src.tts.kernel_backend import install
        from src.tts.weights import load_talker_weights

        print("[server] loading talker weights + installing kernel trunk ...", flush=True)
        weights, _meta = load_talker_weights(MODEL_DIR, device=DEVICE)
        backend, restore = install(tts.model, weights, device=DEVICE)
        print("[server] kernel talker trunk INSTALLED.", flush=True)
    else:
        print("[server] TTS_USE_KERNEL=0 -> running the stock HF talker trunk.", flush=True)

    return tts, restore, backend


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model, restore, backend = await asyncio.to_thread(build_model)
    STATE.update(model=model, restore=restore, backend=backend)
    print("[server] ready.", flush=True)
    try:
        yield
    finally:
        if STATE.get("restore"):
            STATE["restore"]()  # un-swap the trunk (frees nothing critical, but tidy)


app = FastAPI(title="Qwen3-TTS megakernel server", lifespan=lifespan)


# --- request -> generate kwargs ----------------------------------------------
def _gen_kwargs(text: str, language: str | None, ref_audio: str | None,
                ref_text: str | None, do_sample: bool | None) -> dict:
    """Build the kwargs for stream_tts_pcm / generate_voice_clone, with env fallbacks."""
    ref_a = ref_audio or REF_AUDIO
    ref_t = ref_text or REF_TEXT
    if not ref_a or not ref_t:
        raise ValueError(
            "voice-clone model needs ref_audio + ref_text (set TTS_REF_AUDIO/TTS_REF_TEXT "
            "or pass them in the request)"
        )
    return dict(
        generate_method="generate_voice_clone",
        chunk_frames=CHUNK_FRAMES,
        left_context=LEFT_CONTEXT,
        text=text,
        language=(language or LANGUAGE),
        ref_audio=ref_a,
        ref_text=ref_t,
        do_sample=DO_SAMPLE if do_sample is None else do_sample,
    )


async def _pcm_stream(kwargs: dict):
    """Async generator of PCM byte chunks, serialized on the GPU lock."""
    model = STATE["model"]
    if model is None:
        raise RuntimeError("model not loaded")
    async with _GPU_LOCK:
        async for pcm in astream_tts_pcm(model, **kwargs):
            yield pcm


# --- endpoints ---------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "ready": STATE["model"] is not None,
        "model_dir": MODEL_DIR,
        "use_kernel": USE_KERNEL,
        "device": DEVICE,
        "sample_rate": SAMPLE_RATE,
        "format": "pcm_s16le_mono",
        "ref_audio_set": bool(REF_AUDIO),
        "ref_text_set": bool(REF_TEXT),
        "chunk_frames": CHUNK_FRAMES,
        "left_context": LEFT_CONTEXT,
        "do_sample": DO_SAMPLE,
    }


@app.websocket("/tts")
async def tts_ws(ws: WebSocket):
    """Stream PCM frames for one request, then send {"event":"done"}.

    Client -> server (one JSON text message):
        {"text": "...", "language": "english"?, "ref_audio": "..."?, "ref_text": "..."?,
         "do_sample": false?}
    Server -> client: binary PCM frames (int16 LE, 24 kHz mono), then a text frame.
    """
    await ws.accept()
    try:
        req = json.loads(await ws.receive_text())
        kwargs = _gen_kwargs(
            req.get("text", ""), req.get("language"), req.get("ref_audio"),
            req.get("ref_text"), req.get("do_sample"),
        )
        if not kwargs["text"].strip():
            await ws.send_text(json.dumps({"event": "error", "detail": "empty text"}))
            await ws.close()
            return

        n = 0
        async for pcm in _pcm_stream(kwargs):
            await ws.send_bytes(pcm)
            n += len(pcm)
        await ws.send_text(json.dumps({"event": "done", "bytes": n}))
    except WebSocketDisconnect:
        return
    except Exception as e:  # surface the error to the client, then close
        try:
            await ws.send_text(json.dumps({"event": "error", "detail": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.post("/synthesize")
async def synthesize(req: dict):
    """Raw PCM (int16 LE, 24 kHz mono) as a chunked HTTP stream."""
    try:
        kwargs = _gen_kwargs(
            req.get("text", ""), req.get("language"), req.get("ref_audio"),
            req.get("ref_text"), req.get("do_sample"),
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return StreamingResponse(
        _pcm_stream(kwargs),
        media_type="audio/L16",
        headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-Channels": "1"},
    )


def _wav_header(sample_rate: int = SAMPLE_RATE, channels: int = 1, bits: int = 16) -> bytes:
    """Streamable WAV header with unknown length (0xFFFFFFFF sizes)."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


@app.get("/tts.wav")
async def tts_wav(
    text: str = Query(..., description="text to speak"),
    language: str | None = Query(None),
):
    """Convenience: `curl 'http://host:8000/tts.wav?text=hi' -o out.wav`. Streamed WAV."""
    try:
        kwargs = _gen_kwargs(text, language, None, None, None)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    async def gen():
        yield _wav_header()
        async for pcm in _pcm_stream(kwargs):
            yield pcm

    return StreamingResponse(gen(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("TTS_HOST", "0.0.0.0"),
        port=int(os.environ.get("TTS_PORT", "8000")),
    )
