"""
generation/llm_client.py
────────────────────────
vLLM OpenAI-compatible API client for generation.

Design rationale:
- vLLM exposes an OpenAI-compatible HTTP server (`/v1/chat/completions`).
- We use asynchronous HTTP requests (`httpx.AsyncClient`) to avoid blocking 
  the main event loop.
- Supports streaming (token-by-token) to reduce Time-To-First-Token (TTFT)
  perceived by the user.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Dict, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with a vLLM server via the OpenAI-compatible API.
    
    Args:
        base_url: vLLM server URL (e.g. "http://localhost:8000/v1")
        model: Model name configured in vLLM
        max_tokens: Max generated tokens
        temperature: Sampling temperature (low for factual RAG)
        timeout_ms: HTTP timeout in milliseconds
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_tokens: int = 256,
        temperature: float = 0.1,
        timeout_ms: int = 8000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_ms / 1000.0
        
        # Connection pooling across requests
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_s),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )

    async def close(self):
        """Close the HTTP client."""
        await self._http_client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        reraise=True
    )
    async def generate(self, prompt: str) -> str:
        """
        Generate a complete answer (non-streaming).
        
        Args:
            prompt: Full prompt string
            
        Returns:
            Generated text.
        """
        url = f"{self.base_url}/completions"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        
        response = await self._http_client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("text", "")
        return ""

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate an answer token-by-token (streaming).
        Yields tokens as they arrive.
        
        Args:
            prompt: Full prompt string
            
        Yields:
            Generated tokens (strings)
        """
        url = f"{self.base_url}/completions"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        
        try:
            async with self._http_client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                        
                    if line.startswith("data: "):
                        line = line[6:]
                        
                    try:
                        data = json.loads(line)
                        if "choices" in data and len(data["choices"]) > 0:
                            token = data["choices"][0].get("text", "")
                            if token:
                                yield token
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode vLLM stream chunk: {line}")
                        
        except Exception as exc:
            logger.error(f"vLLM streaming error: {exc}")
            raise
