import os
from typing import Dict, Any
from src.retrieval.search import RiskAwareRetriever
from src.guardrails.domain_guard import domain_guard
from src.config.settings import TOP_K_SEARCH

class SingAIRAGPipeline:
    def __init__(self, api_key: str):
        self.retriever = RiskAwareRetriever(api_key=api_key)
        
    def run(self, query: str, lang: str = "en") -> Dict[str, Any]:
        """
        Executes the Two-Stage Risk-Aware RAG pipeline.
        
        Returns:
            Dict containing:
            - decision: "ANSWER", "REFUSE", "ESCALATE"
            - answer: The generated answer or refusal message
            - reason: Logged reason for the decision
            - context: Retrieved context (if any)
        """
        
        # Gate-B: Deterministic Symbolic Guardrails (Pre-retrieval check?)
        # Paper says "Two-Stage Risk-Aware RAG controller"
        # Usually RAG flow: Query -> Retrieve -> Gate-A (Similarity) -> Gate-B (Policy) -> Generate
        # But abstract says:
        # "cosine-calibrated similarity gate for out-of-domain filtering" (Gate-A)
        # "Deterministic Symbolic Guardrails... for detecting credential disclosure..." (Gate-B)
        
        # Check Guardrails first for obvious violations (Input Guardrail)
        is_blocked, reason = domain_guard.domain_guard_action(query)
        if is_blocked:
            return {
                "decision": "REFUSE",
                "answer": "I cannot fulfill this request due to policy constraints.",
                "reason": reason,
                "context": []
            }
            
        # Retrieve
        results = self.retriever.retrieve(query, lang=lang, top_k=TOP_K_SEARCH)
        
        if not results:
             return {
                "decision": "REFUSE",
                "answer": "I do not have enough information to answer this.",
                "reason": "NO_CONTEXT_FOUND",
                "context": []
            }
            
        # Gate-A: Similarity Threshold
        # "similarity thresholding alone effectively filters out-of-domain queries"
        # We check the top score
        top_score = results[0]["score"]
        # Threshold from paper abstract or config? Paper mentions 0.25 in section 3.3
        SIMILARITY_THRESHOLD = 0.25 
        
        if top_score < SIMILARITY_THRESHOLD:
             return {
                "decision": "REFUSE", # Or ESCALATE? Abstract says "out-of-domain filtering"
                "answer": "This query seems outside my knowledge base.",
                "reason": f"LOW_SIMILARITY_SCORE ({top_score:.3f} < {SIMILARITY_THRESHOLD})",
                "context": []
            }
        
        # If passed both, we would Generate.
        # (Generation logic not fully implemented in this repo structure as it requires LLM inference code which was just 'SEA-LION or Qwen' in abstract)
        
        return {
            "decision": "ANSWER",
            "answer": "[GENERATED_ANSWER_PLACEHOLDER]",
            "reason": "PASSED_ALL_GATES",
            "context": results
        }

if __name__ == "__main__":
    # Example usage
    import sys
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("Please set OPENAI_API_KEY")
        sys.exit(1)
        
    pipeline = SingAIRAGPipeline(api_key=key)
    q = "I want a refund for my bill from last month"
    print(f"Query: {q}")
    res = pipeline.run(q, lang="en")
    print(res)
