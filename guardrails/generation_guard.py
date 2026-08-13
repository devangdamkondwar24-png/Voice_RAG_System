"""
guardrails/generation_guard.py
───────────────────────────────
Layer 3 Guardrails: NLI Grounding Check and Citation Validation.

Design rationale:
- Hallucination detection is critical. LLM-as-a-judge is too slow and expensive
  for sub-200ms pipelines.
- We use `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (multilingual NLI) to support
  all 12 Indic languages.
- We employ batched inference across all extracted claims simultaneously to 
  minimize GPU execution time, fitting within the latency budget.
- It treats the retrieved context as the "premise" and the LLM's generated 
  claims as the "hypothesis".
- Output is [Entailment, Neutral, Contradiction]. We require Entailment to
  be the dominant label for a claim to be considered "grounded".
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def _extract_claims_with_citations(text: str) -> List[Dict[str, Any]]:
    """
    Heuristic claim and citation extractor.
    Splits by common sentence endings and extracts [Passage N] citations per claim.
    
    Returns:
        List of dicts: [{"claim": "Clean text", "passage_ranks": [1, 2]}]
    """
    # Split by period, question mark, or Devanagari danda
    raw_claims = re.split(r"[।\.\?]+", text)
    
    claims_data = []
    for rc in raw_claims:
        rc = rc.strip()
        if len(rc) <= 10:
            continue
            
        # Extract citations like [Passage 1]
        matches = re.findall(r"\[Passage (\d+)\]", rc)
        ranks = [int(m) for m in matches]
        
        # Clean claim text (remove the citation markers for the NLI check)
        clean_claim = re.sub(r"\[Passage \d+\]", "", rc).strip()
        
        if len(clean_claim) > 10:
            claims_data.append({
                "claim": clean_claim,
                "passage_ranks": ranks
            })
            
    return claims_data


class GenerationGuard:
    """
    Validates generated text for faithfulness (grounding) using Batched NLI.
    
    Args:
        model_name: Multilingual NLI model
        threshold: Minimum required grounding score (0.0 to 1.0)
        device: "cuda" | "cpu" | "mps"
        batch_size: Batch size for pipeline execution
    """

    def __init__(
        self,
        model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        threshold: float = 0.6,
        device: str = "cuda",
        batch_size: int = 4,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.batch_size = batch_size
        self._pipeline = None

    def _load_model(self) -> None:
        if self._pipeline is not None:
            return
            
        from transformers import pipeline  # type: ignore
        
        logger.info(f"Loading NLI model '{self.model_name}' on {self.device} (batch_size={self.batch_size})…")
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
    ) -> Dict[str, Any]:
        """
        Check if the generated answer is grounded in the provided passages.
        Executes NLI checks in batches.
        
        Returns dict with status, grounding score, and extracted citations.
        """
        if not generated_answer.strip():
             return {
                 "passed": False,
                 "reason": "empty_generation",
                 "grounding_score": 0.0,
                 "citations": [],
                 "total_claims": 0,
                 "entailed_claims": 0,
             }

        claims_data = _extract_claims_with_citations(generated_answer)
        
        # Format citations array as expected by graph state
        citations = []
        for cd in claims_data:
            for rank in cd["passage_ranks"]:
                citations.append({
                    "passage_rank": rank,
                    "claim": cd["claim"]
                })
        
        if not claims_data:
            # If we can't extract claims, assume it's a short/conversational response.
            # Allow it only if it has valid citations.
            has_citations = len(citations) > 0
            return {
                "passed": has_citations,
                "reason": "no_claims_extracted" if not has_citations else None,
                "grounding_score": 1.0 if has_citations else 0.0,
                "citations": citations,
                "total_claims": 0,
                "entailed_claims": 0,
            }

        self._load_model()
        
        # Combine top 3 passages as the premise (context)
        # Using all passages might exceed DeBERTa's context window (512)
        top_context = " ".join([p.get("text", "") for p in retrieved_passages[:3]])
        
        # Build batched inputs: "Premise </s> Hypothesis" (XLM-R / mDeBERTa format)
        pairs = [f"{top_context} </s> {cd['claim']}" for cd in claims_data]
        
        # Execute pipeline in batch
        results_list = self._pipeline(pairs, batch_size=self.batch_size)
        
        entailed_claims = 0
        
        for results in results_list:
            # Flatten if pipeline returned list of lists
            if isinstance(results, list) and isinstance(results[0], list):
                results = results[0]
                
            # Find entailment score
            entailment_score = 0.0
            for r in results:
                label = r["label"].lower()
                # mDeBERTa-v3 uses "entailment"
                if label in ["entailment", "label_0"]:
                    entailment_score = r["score"]
                    break
                    
            if entailment_score > 0.5:
                entailed_claims += 1
                
        grounding_score = entailed_claims / len(claims_data)
        passed = grounding_score >= self.threshold
        
        return {
            "passed": passed,
            "reason": "hallucination_detected" if not passed else None,
            "grounding_score": grounding_score,
            "citations": citations,
            "total_claims": len(claims_data),
            "entailed_claims": entailed_claims,
        }
