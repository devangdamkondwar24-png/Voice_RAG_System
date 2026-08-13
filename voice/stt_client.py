"""
voice/stt_client.py
────────────────────
Sarvam Saaras V3 Realtime WebSocket client.

Design rationale:
- Realtime WebSocket streaming is mandatory for the <200ms perceived latency.
- Client streams audio bytes up to Sarvam, receiving partial transcripts mid-speech.
- Final transcript triggers the RAG pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, Callable, Dict, Optional, Tuple

import websockets
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import get_settings

logger = logging.getLogger(__name__)


class SarvamSTTClient:
    """
    WebSocket client for Sarvam Saaras V3 Realtime STT.
    
    Args:
        api_key: Sarvam API Key
        ws_url: WebSocket URL
        model: Model identifier
        sample_rate: Audio sample rate in Hz
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        ws_url: Optional[str] = None,
        model: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> None:
        settings = get_settings().sarvam
        self.api_key = api_key or settings.api_key
        self.ws_url = ws_url or settings.ws_url
        self.model = model or settings.model
        self.sample_rate = sample_rate

        if not self.api_key:
            logger.warning("SARVAM_API_KEY is not set. STT will fail.")

    @retry(
        retry=retry_if_exception_type((websockets.exceptions.ConnectionClosedError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        reraise=True
    )
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        on_partial: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, str]:
        """
        Stream audio to Sarvam and return the final transcript and language.
        
        Args:
            audio_stream: Async generator yielding audio chunks (PCM 16-bit)
            on_partial: Optional callback fired when partial transcripts arrive
            
        Returns:
            (final_transcript, language_code)
        """
        headers = {"Api-Subscription-Key": self.api_key}
        
        # Build configuration message (first message sent to websocket)
        config_msg = {
            "model": self.model,
            "mode": "transcribe",
            "audio_config": {
                "sample_rate": self.sample_rate,
                "encoding": "linear16",
            },
            "enable_partials": True if on_partial else False,
        }

        final_transcript = ""
        language_code = "en" # Fallback

        try:
            async with websockets.connect(self.ws_url, additional_headers=headers) as ws:
                # 1. Send configuration
                await ws.send(json.dumps(config_msg))
                
                # 2. Wait for server ready ACK
                resp = await ws.recv()
                data = json.loads(resp)
                if data.get("type") == "error":
                    raise RuntimeError(f"Sarvam STT Error: {data.get('message')}")
                
                # 3. Start sender and receiver tasks concurrently
                async def sender():
                    async for chunk in audio_stream:
                        if not chunk:
                            break
                        # For binary data (audio), send as bytes
                        await ws.send(chunk)
                    
                    # Send EOF signal (often an empty binary message or specific JSON)
                    await ws.send(b"")

                async def receiver():
                    nonlocal final_transcript, language_code
                    async for message in ws:
                        if isinstance(message, bytes):
                            continue # Ignore binary responses if any
                            
                        data = json.loads(message)
                        msg_type = data.get("type")
                        
                        if msg_type == "partial":
                            if on_partial:
                                on_partial(data.get("transcript", ""))
                        elif msg_type == "final":
                            final_transcript = data.get("transcript", "")
                            # Saaras detects language; format depends on API exact response
                            language_code = data.get("language", "hi")[:2].lower()
                        elif msg_type == "error":
                            logger.error(f"Sarvam STT Stream Error: {data.get('message')}")
                            break
                        elif msg_type == "eof":
                            break

                await asyncio.gather(sender(), receiver())
                
        except Exception as exc:
            logger.error(f"STT Streaming failed: {exc}")
            raise
            
        return final_transcript.strip(), language_code
