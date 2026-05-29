"""
Phase 4 — code predictor + vocoder + frame-by-frame streaming glue.

The talker (kernel later, HF now) emits one 16-code frame per 12.5 Hz step. The
speech tokenizer's decoder is fully causal and ships a `chunked_decode(codes,
chunk_size, left_context_size)` (CausalConv + causal transformer). We exploit that to
stream: decode small windows of frames with left context, trim the context samples,
emit PCM immediately. NEVER buffer the whole utterance (project constraint).

Pieces:
  - AudioFrameStreamer : forward hook on the talker that pushes each step's 16-code
                         frame onto a queue as it is generated.
  - StreamingVocoder   : incremental wrapper over the decoder's chunked_decode; feed
                         frames, get int16 PCM bytes out, flush at the end.
  - stream_tts_pcm     : runs model.generate in a thread, bridges frames -> vocoder,
                         yields PCM chunks (sync iterator). `astream_tts_pcm` = async.

Audio: 24 kHz, mono, int16 little-endian. ~1920 samples per 12.5 Hz frame.
torch is imported lazily; this file imports without torch installed.
"""

from __future__ import annotations

import queue
import threading

SAMPLE_RATE = 24000
FRAME_HZ = 12.5
SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_HZ)  # 1920
NUM_CODE_GROUPS = 16
CODEC_EOS_ID = 2150
_SENTINEL = object()


# =============================================================================
# 1) Capture per-frame codes from the talker as they are generated
# =============================================================================
class AudioFrameStreamer:
    """Forward hook on Qwen3TTSTalkerForConditionalGeneration.

    Each decode step the talker returns `hidden_states=(layer_hiddens, codec_ids)`
    where codec_ids is the [B, 16] frame (group 0 from the talker, groups 1-15 from
    the code predictor). We push each new frame to a queue for streaming consumption.
    """

    def __init__(self, talker_cg):
        self.q: "queue.Queue" = queue.Queue()
        self._talker = talker_cg
        self._handle = None

    def _hook(self, _module, _args, output):
        codec_ids = None
        hs = getattr(output, "hidden_states", None)
        if isinstance(hs, (tuple, list)) and len(hs) >= 2:
            codec_ids = hs[1]
        if codec_ids is not None:
            # [B,16] -> take batch 0; only single-frame (generate) steps carry codes
            ids = codec_ids.detach().to("cpu").reshape(-1, NUM_CODE_GROUPS)
            self.q.put(ids[-1].tolist())  # latest frame
        return output

    def __enter__(self):
        self._handle = self._talker.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle:
            self._handle.remove()
        self.q.put(_SENTINEL)

    def frames(self):
        """Blocking generator of 16-code frames until the producer signals done."""
        while True:
            item = self.q.get()
            if item is _SENTINEL:
                return
            yield item


# =============================================================================
# 2) Incremental causal vocoder
# =============================================================================
class StreamingVocoder:
    """Incremental wrapper over Qwen3TTSTokenizerV2Decoder.chunked_decode.

    Buffers frames; once `chunk_frames` new frames are available it decodes
    [left_context + chunk] frames, trims the context samples, and returns PCM. Causal
    decoder => trimmed output concatenates seamlessly. Larger left_context = better
    quality but more recompute per chunk (RTF cost); first chunk has no context so no
    added latency.
    """

    def __init__(self, decoder, *, samples_per_frame=SAMPLES_PER_FRAME,
                 chunk_frames=6, left_context=25, device="cuda"):
        self.decoder = decoder
        self.spf = samples_per_frame
        self.chunk_frames = chunk_frames
        self.left_context = left_context
        self.device = device
        self._buf = []        # all frames seen (kept for left context)
        self._emitted = 0      # number of frames already turned into output

    def _decode_range(self, start, end):
        import torch
        ctx = min(self.left_context, start)
        codes = self._buf[start - ctx:end]                  # list of [16]
        t = torch.tensor(codes, device=self.device).t().unsqueeze(0)  # [1,16,L]
        t = torch.clamp(t, min=0)
        with torch.no_grad():
            wav = self.decoder(t).squeeze(0).squeeze(0)     # [L*upsample]
        wav = wav[ctx * self.spf:]                           # drop context samples
        return _to_pcm16(wav)

    def feed(self, frame):
        """Add one 16-code frame; return PCM bytes if a chunk is ready, else b''."""
        self._buf.append(frame)
        ready = len(self._buf) - self._emitted
        if ready >= self.chunk_frames:
            start, end = self._emitted, len(self._buf)
            self._emitted = end
            return self._decode_range(start, end)
        return b""

    def flush(self):
        """Decode any remaining buffered frames."""
        if self._emitted >= len(self._buf):
            return b""
        start, end = self._emitted, len(self._buf)
        self._emitted = end
        return self._decode_range(start, end)


def _to_pcm16(wav):
    import numpy as np
    import torch
    a = wav.detach().to("cpu", dtype=torch.float32).numpy()
    a = np.clip(a, -1.0, 1.0)
    return (a * 32767.0).astype("<i2").tobytes()


# =============================================================================
# 3) End-to-end streaming
# =============================================================================
def stream_tts_pcm(model, *, generate_method="generate_voice_clone",
                   chunk_frames=6, left_context=25, **gen_args):
    """Yield 24 kHz mono int16 PCM chunks as the talker generates, frame by frame.

    `model` is the qwen_tts Qwen3TTSModel wrapper. `generate_method` selects the helper
    ("generate_voice_clone" for *-Base, "generate_custom_voice"/"generate_voice_design"
    for variants with named speakers); `gen_args` are forwarded to it (e.g. text=...,
    language=..., ref_audio=..., ref_text=...). Runs generate() in a worker thread; a
    forward hook feeds per-frame codes to the vocoder; this generator yields PCM.

    Attribute paths confirmed on Colab: the qwen_tts wrapper holds the HF model on
    `.model`; the speech tokenizer is `hf.speech_tokenizer` (a Qwen3TTSTokenizer wrapper)
    whose raw V2 model is `.model`, holding `.decoder` and `get_decode_upsample_rate()`.
    """
    hf = model.model                       # Qwen3TTSForConditionalGeneration
    talker_cg = hf.talker                  # Qwen3TTSTalkerForConditionalGeneration
    decoder = hf.speech_tokenizer.model.decoder       # Qwen3TTSTokenizerV2Decoder
    spf = int(hf.speech_tokenizer.get_decode_upsample_rate())  # samples/frame (1920)
    device = next(hf.parameters()).device

    voc = StreamingVocoder(decoder, samples_per_frame=spf, chunk_frames=chunk_frames,
                           left_context=left_context, device=device)
    streamer = AudioFrameStreamer(talker_cg)

    gen = getattr(model, generate_method)
    err = {}

    def _run():
        try:
            gen(**gen_args)
        except Exception as e:  # surface to main thread
            err["e"] = e
        finally:
            # signal end-of-frames so the consume loop below terminates (generate
            # produced its last frame). Without this the loop blocks forever.
            streamer.q.put(_SENTINEL)

    with streamer:
        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        for frame in streamer.frames():
            if frame and frame[0] == CODEC_EOS_ID:
                break
            pcm = voc.feed(frame)
            if pcm:
                yield pcm
        worker.join()
    tail = voc.flush()
    if tail:
        yield tail
    if "e" in err:
        raise err["e"]


async def astream_tts_pcm(model, **kwargs):
    """Async wrapper for Pipecat: yields PCM chunks off a worker thread."""
    import asyncio

    q: "asyncio.Queue" = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _producer():
        try:
            for pcm in stream_tts_pcm(model, **kwargs):
                loop.call_soon_threadsafe(q.put_nowait, pcm)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

    threading.Thread(target=_producer, daemon=True).start()
    while True:
        pcm = await q.get()
        if pcm is _SENTINEL:
            return
        yield pcm
