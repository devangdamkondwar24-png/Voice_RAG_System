"""
generation/prompts.py
──────────────────────
Grounded RAG prompt templates.

Design rationale:
- The LLM MUST be instructed to answer ONLY from the provided passages.
- It MUST cite the passages using the [Passage N] format to allow the 
  GenerationGuard to validate the citations.
- The prompt includes the human-readable name of the target language
  to ensure the LLM responds in the correct language.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from ingestion.dataset_loader import LANG_NAMES

logger = logging.getLogger(__name__)


def build_rag_prompt(
    query: str,
    language_code: str,
    passages: List[Dict],
) -> str:
    """
    Build the full prompt string for the LLM.
    
    Args:
        query: User's question
        language_code: ISO 639-1 code (e.g. "hi")
        passages: List of Qdrant payloads (parent chunks from retrieval)
        
    Returns:
        Formatted prompt string.
    """
    language_name = LANG_NAMES.get(language_code, "English")
    
    # Format passages with [Passage N] tags
    formatted_passages = []
    for i, p in enumerate(passages, 1):
        text = p.get("text", "").strip()
        if text:
            formatted_passages.append(f"[Passage {i}]\n{text}")
            
    context_str = "\n\n".join(formatted_passages)
    
    prompt = f"""You are a helpful, factual, and multilingual information assistant.
Your task is to answer the user's question using ONLY the provided passages.

Language: {language_code}
Answer in: {language_name}

=== PASSAGES ===
{context_str}
================

Question: {query}

RULES:
1. Answer ONLY based on the passages above. Do not use outside knowledge.
2. You MUST cite your sources for every claim using the exact format [Passage N].
3. If the passages do not contain enough information to answer the question, state that you cannot answer based on the provided context.
4. Respond entirely in {language_name}.
5. Keep your answer clear and concise (under 200 words).

Answer in {language_name}:
"""
    return prompt
