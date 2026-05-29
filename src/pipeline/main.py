"""
Full Pipecat voice pipeline:  mic -> STT -> LLM -> TTS (megakernel) -> speaker.

This is the conversational loop. The TTS leg is our megakernel server, wired in via
`QwenMegakernelTTSService` (src/tts/service.py). The STT and LLM legs use off-the-shelf
providers and therefore need API keys — that's why the deterministic interview demo is
`scripts/say.py` (text -> audio, no keys), and this is the "full agent" path on top.

Run (laptop or the GPU box; the TTS server must be reachable):
    pip install "pipecat-ai[deepgram,openai,local]"          # or your chosen providers
    export DEEPGRAM_API_KEY=...        # STT
    export OPENAI_API_KEY=...          # LLM   (or swap in Anthropic below)
    export TTS_WS_URL=ws://127.0.0.1:8000/tts
    export TTS_REF_AUDIO=ref.wav TTS_REF_TEXT="..."   # if server has no defaults
    python src/pipeline/main.py

Provider choices here (Deepgram STT, OpenAI LLM, LocalAudioTransport) are swappable —
they're the off-the-shelf parts. The only project-specific piece is the TTS service.
Pipecat's API moves between releases; if imports fail, the guard prints what to install.
This is prep scaffolding: confirm against your installed pipecat version before relying
on it. See docs/phase5-runbook.md.
"""

from __future__ import annotations

import asyncio
import os
import sys

# repo root on path so `src.*` resolves when run as `python src/pipeline/main.py`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _import_pipecat():
    """Import pipecat pieces, returning a namespace dict. Raises with guidance if absent."""
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.services.deepgram.stt import DeepgramSTTService
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.transports.local.audio import (
            LocalAudioTransport,
            LocalAudioTransportParams,
        )

        from src.tts.service import QwenMegakernelTTSService

        return dict(
            Pipeline=Pipeline,
            PipelineRunner=PipelineRunner,
            PipelineTask=PipelineTask,
            PipelineParams=PipelineParams,
            SileroVADAnalyzer=SileroVADAnalyzer,
            DeepgramSTTService=DeepgramSTTService,
            OpenAILLMService=OpenAILLMService,
            LocalAudioTransport=LocalAudioTransport,
            LocalAudioTransportParams=LocalAudioTransportParams,
            QwenMegakernelTTSService=QwenMegakernelTTSService,
        )
    except ImportError as e:
        raise SystemExit(
            f"pipecat import failed ({e}).\n"
            "Install: pip install \"pipecat-ai[deepgram,openai,local,silero]\"\n"
            "Import paths vary by pipecat version — adjust _import_pipecat() to match yours."
        )


async def main():
    P = _import_pipecat()

    transport = P["LocalAudioTransport"](
        P["LocalAudioTransportParams"](
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=P["SileroVADAnalyzer"](),
        )
    )

    stt = P["DeepgramSTTService"](api_key=os.environ["DEEPGRAM_API_KEY"])

    llm = P["OpenAILLMService"](
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    )

    tts = P["QwenMegakernelTTSService"](
        ws_url=os.environ.get("TTS_WS_URL", "ws://127.0.0.1:8000/tts"),
        language=os.environ.get("TTS_LANGUAGE", "english"),
        ref_audio=os.environ.get("TTS_REF_AUDIO") or None,
        ref_text=os.environ.get("TTS_REF_TEXT") or None,
    )

    # Conversation context: seed a system prompt, aggregate user+assistant turns.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful voice assistant. Keep replies short and spoken-friendly.",
        }
    ]
    context = llm.create_context_aggregator  # alias check below

    # Newer pipecat: OpenAILLMContext + llm.create_context_aggregator(context)
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

    ctx = OpenAILLMContext(messages)
    aggregator = context(ctx)

    pipeline = P["Pipeline"](
        [
            transport.input(),       # mic
            stt,                     # speech -> text
            aggregator.user(),       # add user turn to context
            llm,                     # text -> reply text
            tts,                     # reply text -> megakernel audio
            transport.output(),      # speaker
            aggregator.assistant(),  # add assistant turn to context
        ]
    )

    task = P["PipelineTask"](
        pipeline,
        params=P["PipelineParams"](allow_interruptions=True),
    )

    print("[pipeline] talk into the mic; Ctrl-C to stop.")
    await P["PipelineRunner"]().run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[pipeline] stopped.")
