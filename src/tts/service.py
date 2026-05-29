"""
Pipecat-compatible TTS service backed by the Qwen3-TTS megakernel server.

Connects to the FastAPI WebSocket endpoint (src/server/main.py, `/tts`), streams PCM
frames back, and re-emits them as Pipecat `TTSAudioRawFrame`s as they arrive — so the
voice pipeline starts speaking on the first chunk (no full-utterance buffering).

Two reasons to talk to the server over WS rather than holding the model in-process:
  - the GPU + kernel live in exactly one place (the server owns batch-1 serialization);
  - the pipeline process can run anywhere (laptop), not just on the 5090 box.

Pipecat import paths moved between releases, so they're resolved defensively below. If
your installed pipecat differs, this is the one spot to adjust (prep code — see
docs/phase5-runbook.md). Targeted at pipecat-ai's TTSService contract:
`run_tts(text) -> async-yields frames`, framed by TTSStartedFrame / TTSStoppedFrame.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

# --- pipecat imports (version-tolerant) --------------------------------------
try:  # newer layout
    from pipecat.services.tts_service import TTSService
except ImportError:  # older layout
    from pipecat.services.ai_services import TTSService  # type: ignore

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

SAMPLE_RATE = 24000


class QwenMegakernelTTSService(TTSService):
    """Streams 24 kHz mono PCM from the megakernel server over WebSocket.

    Args:
        ws_url:    e.g. "ws://127.0.0.1:8000/tts"
        language:  Qwen language tag (lowercase, e.g. "english")
        ref_audio: path to the reference clip ON THE SERVER (voice-clone model). May be
                   omitted if the server has TTS_REF_AUDIO/TTS_REF_TEXT set.
        ref_text:  transcript of ref_audio.
    """

    def __init__(
        self,
        *,
        ws_url: str = "ws://127.0.0.1:8000/tts",
        language: str = "english",
        ref_audio: str | None = None,
        ref_text: str | None = None,
        sample_rate: int = SAMPLE_RATE,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._ws_url = ws_url
        self._language = language
        self._ref_audio = ref_audio
        self._ref_text = ref_text

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        import websockets  # lazy: only needed when the pipeline actually runs

        req: dict = {"text": text, "language": self._language}
        if self._ref_audio:
            req["ref_audio"] = self._ref_audio
        if self._ref_text:
            req["ref_text"] = self._ref_text

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()
        try:
            async with websockets.connect(self._ws_url, max_size=None) as ws:
                await ws.send(json.dumps(req))
                async for msg in ws:
                    if isinstance(msg, (bytes, bytearray)):
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(
                            audio=bytes(msg),
                            sample_rate=self.sample_rate,
                            num_channels=1,
                        )
                    else:  # terminal text frame: {"event": "done"|"error"}
                        evt = json.loads(msg)
                        if evt.get("event") == "error":
                            yield ErrorFrame(f"TTS server error: {evt.get('detail')}")
                        break
        except Exception as e:
            yield ErrorFrame(f"TTS connection failed: {e}")
        finally:
            yield TTSStoppedFrame()
