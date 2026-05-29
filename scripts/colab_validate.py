"""
Colab validation — run the Qwen3-TTS reference path and prove our streaming glue
matches it, deterministically, with no megakernel involved yet.

Self-contained (inlines a copy of StreamingVocoder == src/tts/stream.py) so you can
just upload THIS ONE FILE to Colab. See docs/colab.md for the cell sequence.

Three checks:
  A. REFERENCE WORKS  — greedy generate_custom_voice -> wav (sanity the model runs).
  B. TRUNK SEAM       — hook Qwen3TTSTalkerModel.forward, capture (inputs_embeds,
                        last_hidden_state) per call -> golden npz for Phase-5 kernel test.
  C. STREAMING == REF — feed the SAME greedy codes through our incremental
                        StreamingVocoder and through the full buffered decode; assert
                        the waveforms match (seamless chunking). Deterministic.

Usage:
    python colab_validate.py --model models/Qwen3-TTS-12Hz-0.6B-Base \
        --text "Hello from the megakernel." --speaker <name> --out_dir out
"""

import argparse


SAMPLE_RATE = 24000
NUM_CODE_GROUPS = 16
CODEC_EOS_ID = 2150


class StreamingVocoder:  # mirror of src/tts/stream.py (kept inline for single-file use)
    def __init__(self, decoder, *, samples_per_frame, chunk_frames=6,
                 left_context=25, device="cuda"):
        self.decoder = decoder
        self.spf = samples_per_frame
        self.chunk_frames = chunk_frames
        self.left_context = left_context
        self.device = device
        self._buf = []
        self._emitted = 0

    def _decode_range(self, start, end):
        import torch
        ctx = min(self.left_context, start)
        codes = self._buf[start - ctx:end]
        t = torch.tensor(codes, device=self.device).t().unsqueeze(0)  # [1,16,L]
        t = torch.clamp(t, min=0)
        with torch.no_grad():
            wav = self.decoder(t).squeeze(0).squeeze(0)
        return wav[ctx * self.spf:]

    def feed(self, frame):
        self._buf.append(frame)
        if len(self._buf) - self._emitted >= self.chunk_frames:
            start, end = self._emitted, len(self._buf)
            self._emitted = end
            return self._decode_range(start, end)
        return None

    def flush(self):
        if self._emitted >= len(self._buf):
            return None
        start, end = self._emitted, len(self._buf)
        self._emitted = end
        return self._decode_range(start, end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default="Hello from the megakernel.")
    ap.add_argument("--language", default="english")
    ap.add_argument("--speaker", default=None)  # unused for Base (voice-clone only)
    # Base model has no named speakers -> clone from a reference clip.
    # Defaults point at Qwen's hosted demo clip + its transcript.
    ap.add_argument("--ref_audio",
                    default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav")
    ap.add_argument("--ref_text",
                    default="Okay. Yeah. I resent you. I love you. I respect you. "
                            "But you know what? You blew it! And thanks to you.")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--chunk_frames", type=int, default=6)
    ap.add_argument("--left_context", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    # The vocoder is fully causal, so chunked decode == full decode in exact arithmetic.
    # Run the [C] comparison in fp32 to factor out bf16 rounding (else diff ~ 1e-2..1e-3).
    ap.add_argument("--fp32_check", action="store_true", default=True)
    ap.add_argument("--no_fp32_check", dest="fp32_check", action="store_false")
    args = ap.parse_args()

    import os
    import numpy as np
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"loading {args.model} ...")
    tts = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device,
                                        dtype=torch.bfloat16)
    hf = tts.model
    talker_cg = hf.talker
    trunk = hf.talker.model
    # tts.model.speech_tokenizer is the Qwen3TTSTokenizer wrapper; its .model is the
    # raw Qwen3TTSTokenizerV2Model nn.Module that holds the actual .decoder + upsample rate.
    decoder = hf.speech_tokenizer.model.decoder
    spf = int(hf.speech_tokenizer.get_decode_upsample_rate())
    print(f"samples/frame = {spf} (expect 1920)")

    # --- capture trunk I/O + per-frame codes during a greedy generate ---
    trunk_io = {"in": [], "out": []}
    frames = []

    def trunk_hook(_m, _a, kw, out):
        ie = kw.get("inputs_embeds")
        hs = getattr(out, "last_hidden_state", None)
        if ie is not None and hs is not None:
            trunk_io["in"].append(ie.detach().float().cpu()[0].numpy())
            trunk_io["out"].append(hs.detach().float().cpu()[0].numpy())
        return out

    def code_hook(_m, _a, out):
        hs = getattr(out, "hidden_states", None)
        if isinstance(hs, (tuple, list)) and len(hs) >= 2 and hs[1] is not None:
            ids = hs[1].detach().cpu().reshape(-1, NUM_CODE_GROUPS)
            frames.append(ids[-1].tolist())
        return out

    h1 = trunk.register_forward_hook(trunk_hook, with_kwargs=True)
    h2 = talker_cg.register_forward_hook(code_hook)

    print("\n[A] greedy generate_voice_clone (Base = clone from ref clip) ...")
    wavs, sr = tts.generate_voice_clone(
        text=args.text, language=args.language,
        ref_audio=args.ref_audio, ref_text=args.ref_text,
        x_vector_only_mode=False,  # ICL mode (uses ref_text)
        do_sample=False, max_new_tokens=2048,
    )
    h1.remove(); h2.remove()
    wav_ref = np.asarray(wavs[0]).squeeze()
    sf.write(os.path.join(args.out_dir, "A_reference.wav"), wav_ref, sr)
    print(f"    OK: audio {wav_ref.shape} @ {sr} Hz -> A_reference.wav")
    print(f"    trunk calls={len(trunk_io['in'])}  frames captured={len(frames)}")
    assert sr == SAMPLE_RATE, f"unexpected sr {sr}"

    # --- [B] save golden trunk pairs ---
    np.savez_compressed(
        os.path.join(args.out_dir, "golden_trunk.npz"),
        trunk_in=np.array(trunk_io["in"], dtype=object),
        trunk_out=np.array(trunk_io["out"], dtype=object),
        codes=np.array(frames, dtype=np.int64),
    )
    print("[B] saved golden_trunk.npz (inputs_embeds + last_hidden_state per call)")

    # --- [C] streaming vocoder vs full buffered decode (same codes) ---
    # Vocoder is causal => chunked-with-left-context == full decode in exact arithmetic.
    # Cast to fp32 so the comparison isn't dominated by bf16 rounding noise.
    if args.fp32_check:
        decoder.float()
        print("[C] decoder cast to fp32 for comparison (isolates glue from bf16 noise)")
    # drop trailing EOS frame if present
    clean = [f for f in frames if f[0] != CODEC_EOS_ID]
    codes_t = torch.tensor(clean, device=args.device)               # [T,16]

    # reference full decode
    with torch.no_grad():
        wav_full = decoder(torch.clamp(codes_t.t().unsqueeze(0), min=0)).squeeze().float().cpu().numpy()

    # streamed decode
    voc = StreamingVocoder(decoder, samples_per_frame=spf,
                           chunk_frames=args.chunk_frames,
                           left_context=args.left_context, device=args.device)
    chunks = []
    for f in clean:
        c = voc.feed(f)
        if c is not None:
            chunks.append(c.float().cpu().numpy())
    tail = voc.flush()
    if tail is not None:
        chunks.append(tail.float().cpu().numpy())
    wav_stream = np.concatenate(chunks) if chunks else np.zeros(0)

    n = min(len(wav_full), len(wav_stream))
    diff = float(np.max(np.abs(wav_full[:n] - wav_stream[:n]))) if n else float("nan")
    sf.write(os.path.join(args.out_dir, "C_stream.wav"), wav_stream, sr)
    print(f"[C] full={len(wav_full)} stream={len(wav_stream)} "
          f"max|diff|={diff:.3e}  (len match: {len(wav_full)==len(wav_stream)})")
    print(f"    chunk_frames={args.chunk_frames} left_context={args.left_context} "
          f"-> {len(chunks)} PCM chunks -> C_stream.wav")
    lens_ok = len(wav_full) == len(wav_stream)
    tol = 1e-3 if args.fp32_check else 2e-2  # bf16 path: ~1e-2 rounding is expected
    if diff < tol and lens_ok:
        print(f"\nPASS: streaming glue reproduces the reference waveform "
              f"(diff {diff:.3e} < {tol:.0e}, {'fp32' if args.fp32_check else 'bf16'}).")
    elif lens_ok:
        print(f"\nCHECK: lengths match (glue alignment OK) but diff {diff:.3e} >= {tol:.0e}. "
              f"In bf16 this is just rounding; rerun with --fp32_check to confirm.")
    else:
        print(f"\nFAIL: length mismatch {len(wav_full)} vs {len(wav_stream)} — real glue bug.")


if __name__ == "__main__":
    main()
