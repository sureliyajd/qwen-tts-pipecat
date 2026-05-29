"""
CLI demo client for the Qwen3-TTS megakernel server (Option A — no mic/STT/LLM needed).

Sends text to the running server, streams the PCM back frame by frame, writes a WAV and
(optionally) plays it live. This is the deterministic interview demo: text in -> spoken
audio out, straight off the megakernel, with the time-to-first-chunk printed.

    # server must be running (see src/server/main.py)
    python scripts/say.py "Hello from the megakernel." -o out.wav
    python scripts/say.py "Hello from the megakernel." --play     # live playback
    python scripts/say.py "..." --host 1.2.3.4 --port 8000

Deps: `websockets` (required). `sounddevice` + `numpy` only if you pass --play.
Everything else is stdlib. Audio: 24 kHz mono int16.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time

SAMPLE_RATE = 24000


def _wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE, channels: int = 1, bits: int = 16) -> bytes:
    """Wrap raw int16 PCM in a complete WAV container."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_len = len(pcm)
    return (
        b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", data_len) + pcm
    )


async def run(args) -> int:
    try:
        import websockets
    except ImportError:
        print("error: `pip install websockets` required for this client", file=sys.stderr)
        return 2

    player = None
    if args.play:
        try:
            import numpy as np
            import sounddevice as sd
            player = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
            player.start()
            _np = np
        except Exception as e:
            print(f"warning: --play unavailable ({e}); writing file only", file=sys.stderr)
            player = None

    uri = f"ws://{args.host}:{args.port}/tts"
    req = {"text": args.text, "language": args.language}
    if args.do_sample:
        req["do_sample"] = True

    chunks: list[bytes] = []
    t0 = time.perf_counter()
    ttfc = None

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps(req))
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                if ttfc is None:
                    ttfc = (time.perf_counter() - t0) * 1000.0
                    print(f"  TTFC (first chunk): {ttfc:.1f} ms")
                chunks.append(bytes(msg))
                if player is not None:
                    player.write(_np.frombuffer(msg, dtype="<i2"))
            else:  # text frame == terminal event
                evt = json.loads(msg)
                if evt.get("event") == "error":
                    print(f"server error: {evt.get('detail')}", file=sys.stderr)
                    return 1
                break

    if player is not None:
        player.stop()
        player.close()

    pcm = b"".join(chunks)
    secs = len(pcm) / 2 / SAMPLE_RATE
    wall = time.perf_counter() - t0
    rtf = wall / secs if secs else float("inf")
    print(f"  audio: {secs:.2f}s  |  wall: {wall:.2f}s  |  RTF: {rtf:.3f}  |  bytes: {len(pcm)}")

    if args.out:
        with open(args.out, "wb") as f:
            f.write(_wav_bytes(pcm))
        print(f"  wrote {args.out}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Qwen3-TTS megakernel demo client")
    p.add_argument("text", help="text to speak")
    p.add_argument("-o", "--out", default="out.wav", help="output WAV path (default out.wav)")
    p.add_argument("--play", action="store_true", help="play audio live (needs sounddevice)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--language", default="english")
    p.add_argument("--do-sample", dest="do_sample", action="store_true",
                   help="sampled voice instead of greedy")
    args = p.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
