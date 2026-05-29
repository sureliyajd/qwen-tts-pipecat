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

        # VAD is optional: silero can drag in torch (no wheel on py3.14). If absent we
        # run without a transport-level VAD and lean on Deepgram's endpointing.
        try:
            from pipecat.audio.vad.silero import SileroVADAnalyzer
        except Exception:
            SileroVADAnalyzer = None

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
            "Install: pip install \"pipecat-ai[deepgram,openai,local]\"\n"
            "Import paths vary by pipecat version — adjust _import_pipecat() to match yours."
        )


def _build_demo_tap():
    """A tidy on-screen transcript tap: prints '🎤 You:' / '🤖 Bot:' as turns happen.

    Two instances are placed in the pipeline (after STT for user text, after the LLM for
    assistant text). Forwards every frame unchanged.
    """
    from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
    from pipecat.frames.frames import (
        TranscriptionFrame,
        LLMTextFrame,
        LLMFullResponseEndFrame,
    )

    class DemoTap(FrameProcessor):
        def __init__(self, role):
            super().__init__()
            self._role = role
            self._buf = ""

        async def process_frame(self, frame, direction: "FrameDirection"):
            await super().process_frame(frame, direction)
            if self._role == "user" and isinstance(frame, TranscriptionFrame):
                if frame.text and frame.text.strip():
                    print(f"\n🎤 You:  {frame.text.strip()}", flush=True)
            elif self._role == "bot":
                if isinstance(frame, LLMTextFrame):
                    self._buf += frame.text
                elif isinstance(frame, LLMFullResponseEndFrame) and self._buf.strip():
                    print(f"🤖 Bot:  {self._buf.strip()}", flush=True)
                    self._buf = ""
            await self.push_frame(frame, direction)

    return DemoTap("user"), DemoTap("bot")


async def main():
    # Quiet Pipecat's DEBUG/INFO spam so the demo terminal shows a clean transcript.
    import sys as _sys
    try:
        from loguru import logger as _lg
        _lg.remove()
        _lg.add(_sys.stderr, level="WARNING")
    except Exception:
        pass

    P = _import_pipecat()

    # Longer stop_secs so a normal pause mid-sentence doesn't end the turn (fixes the
    # "fragmented into many short turns" behavior). Tune via VAD_STOP_SECS.
    vad = None
    if P["SileroVADAnalyzer"]:
        try:
            from pipecat.audio.vad.vad_analyzer import VADParams
            stop_secs = float(os.environ.get("VAD_STOP_SECS", "1.2"))
            vad = P["SileroVADAnalyzer"](params=VADParams(stop_secs=stop_secs))
        except Exception:
            vad = P["SileroVADAnalyzer"]()
    else:
        print("[pipeline] no silero VAD installed — relying on Deepgram endpointing.")
    transport = P["LocalAudioTransport"](
        P["LocalAudioTransportParams"](
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=vad,
        )
    )

    stt = P["DeepgramSTTService"](api_key=os.environ["DEEPGRAM_API_KEY"])

    # LLM: Groq (free, OpenAI-compatible endpoint) if GROQ_API_KEY set, else OpenAI.
    if os.environ.get("GROQ_API_KEY"):
        llm = P["OpenAILLMService"](
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            model=os.environ.get("LLM_MODEL", "llama-3.1-8b-instant"),
        )
    else:
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
            "content": "You are a voice assistant. Reply in ONE sentence, "
                       "max 8 words. Be direct. No preamble, no emojis.",
        }
    ]
    # pipecat 1.3.x: universal LLMContext + LLMContextAggregatorPair.
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
    )

    ctx = LLMContext(messages)
    aggregator = LLMContextAggregatorPair(ctx)

    user_tap, bot_tap = _build_demo_tap()

    pipeline = P["Pipeline"](
        [
            transport.input(),       # mic
            stt,                     # speech -> text
            user_tap,                # print "You: ..."
            aggregator.user(),       # add user turn to context
            llm,                     # text -> reply text
            bot_tap,                 # print "Bot: ..."
            tts,                     # reply text -> megakernel audio
            transport.output(),      # speaker
            aggregator.assistant(),  # add assistant turn to context
        ]
    )

    task = P["PipelineTask"](
        pipeline,
        params=P["PipelineParams"](allow_interruptions=True),
    )

    print("\n" + "=" * 56)
    print("  Qwen3-TTS megakernel voice agent — LIVE")
    print("  mic -> Deepgram STT -> Groq LLM -> megakernel TTS -> speaker")
    print("  Talk now. Ctrl-C to stop.")
    print("=" * 56, flush=True)
    await P["PipelineRunner"]().run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[pipeline] stopped.")
