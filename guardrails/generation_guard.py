"""
guardrails/generation_guard.py
───────────────────────────────
Layer 3 Guardrails: NLI Grounding Check and Citation Validation.

Design rationale:
- Hallucination detection is critical. LLM-as-a-judge is too slow and expensive
  for sub-200ms pipelines.
- We use `DeBERTa-v3` fine-tuned on NLI (Natural Language Inference).
- It treats the retrieved context as the "premise" and the LLM's generated 
  claims as the "hypothesis".
- Output is [Entailment, Neutral, Contradiction]. We require Entailment to
  be the dominant label for a claim to be considered "grounded".
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _extract_claims(text: str) -> List[str]:
    """
    Very simple heuristic claim extractor.
    Splits by common sentence endings and strips citations.
    """
    # Remove citations like [Passage 1]
    text_clean = re.sub(r"\[Passage \d+\]", "", text)
    # Split by period, question mark, or Devanagari danda
    raw_claims = re.split(r"[।\.\?]+", text_clean)
    return [c.strip() for c in raw_claims if len(c.strip()) > 10]


def _extract_citations(text: str) -> List[str]:
    """Extract all [Passage N] citations from text."""
    matches = re.findall(r"\[Passage (\d+)\]", text)
    return [f"Passage {m}" for m in matches]


class GenerationGuard:
    """
    Validates generated text for faithfulness (grounding).
    
    Args:
        model_name: DeBERTa NLI model
        threshold: Minimum required grounding score (0.0 to 1.0)
        device: "cuda" | "cpu" | "mps"
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        threshold: float = 0.6,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self._pipeline = None

    def _load_model(self) -> None:
        if self._pipeline is not None:
            return
            
        from transformers import pipeline  # type: ignore
        
        logger.info(f"Loading NLI model '{self.model_name}' on {self.device}…")
        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            device=self.device,
            # Return all scores to inspect entailment vs contradiction
            top_k=None,
        )

    def check_grounding(
        self,
        generated_answer: str,
        retrieved_passages: List[Dict],
    ) -> Dict[str, any]:
        """
        Check if the generated answer is grounded in the provided passages.
        
        Returns dict with status and grounding score.
        """
        if not generated_answer.strip():
             return {
                 "passed": False,
                 "reason": "empty_generation",
                 "grounding_score": 0.0,
                 "citations_valid": False,
             }

        claims = _extract_claims(generated_answer)
        citations = _extract_citations(generated_answer)
        
        if not claims:
            # If we can't extract claims, we assume it's a short/conversational response
            # which we allow if it cites something.
            return {
                "passed": len(citations) > 0,
                "reason": "no_claims_extracted" if not citations else None,
                "grounding_score": 1.0 if citations else 0.0,
                "citations_valid": len(citations) > 0,
            }

        self._load_model()
        
        # Combine top 3 passages as the premise (context)
        # Using all passages might exceed DeBERTa's context window (512)
        top_context = " ".join([p.get("text", "") for p in retrieved_passages[:3]])
        
        entailed_claims = 0
        
        for claim in claims:
            # Format: "Premise [SEP] Hypothesis"
            pair = f"{top_context} </s> {claim}"
            # Pipeline returns list of dicts: [{'label': 'entailment', 'score': 0.9}, ...]
            results = self._pipeline(pair)
            
            # Find entailment score
            entailment_score = 0.0
            for r in results:
                if r["label"].lower() in ["entailment", "label_0"]: # DeBERTa-v3 usually maps label_0 to entailment
                    entailment_score = r["score"]
                    break
                    
            if entailment_score > 0.5:
                entailed_claims += 1
                
        grounding_score = entailed_claims / len(claims)
        passed = grounding_score >= self.threshold
        
        return {
            "passed": passed,
            "reason": "hallucination_detected" if not passed else None,
            "grounding_score": grounding_score,
            "citations_valid": len(citations) > 0,
            "total_claims": len(claims),
            "entailed_claims": entailed_claims,
        }
