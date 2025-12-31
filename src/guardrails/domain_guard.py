import re
from typing import Tuple, Optional

# --- Guardrail Patterns ---
INJECTION_PATTERNS = [
    re.compile(r"\b(ignore|disregard|override)\b.*\b(instructions|rules)\b", re.I),
    re.compile(r"\b(system\s+prompt|developer\s+message|hidden\s+prompt)\b", re.I),
    re.compile(r"\b(jailbreak|dan mode)\b", re.I),
]

DISCLOSE_VERBS = [
    re.compile(r"\b(show|give|display|reveal|list)\b", re.I),
    re.compile(r"\b(what|where)\b.*\b(is|are)\b", re.I),
]

SECRET_NOUNS = [
    re.compile(r"\b(password|credential|key|token|secret|private)\b", re.I),
    re.compile(r"\b(system\s+prompt)\b", re.I),
]

class DomainGuard:
    def __init__(self):
        self.injection_patterns = INJECTION_PATTERNS
        # Add others if needed based on paper details

    def check_injection(self, query: str) -> bool:
        for p in self.injection_patterns:
            if p.search(query):
                return True
        return False

    def domain_guard_action(self, query: str, refund_window_days: int = 7, refund_duplicate_only: bool = True) -> Tuple[bool, str]:
        """
        Gate-B: Deterministic Symbolic Guardrails.
        Returns (is_blocked, reason).
        """
        # 1. Prompt Injection
        if self.check_injection(query):
            return True, "BLOCKED_INJECTION"

        # 2. Credential Disclosure (Heuristic reconstruction)
        # Detailed logic implied from abstract: check patterns for secrets
        for noun_re in SECRET_NOUNS:
            if noun_re.search(query):
                 for verb_re in DISCLOSE_VERBS:
                     if verb_re.search(query):
                         return True, "BLOCKED_CREDENTIAL_DISCLOSURE"
        
        # 3. Policy Constraints (simple keyword checks for demonstration)
        # Example: "refund" + "last month" > 7 days
        # This part requires more sophisticated parsing or the exact Colab logic.
        # Implemented as a basic keyword check for now.
        q_lower = query.lower()
        if "refund" in q_lower:
             if "last month" in q_lower or "3 months" in q_lower:
                return True, "BLOCKED_POLICY_REFUND_WINDOW"

        return False, ""

# Singleton instance
domain_guard = DomainGuard()
