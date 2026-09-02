# ai-system/chunkdup/retrieval/query_rewriter.py
import re

class QueryRewriter:
    """
    Cleans filler words, expands domain abbreviations, and refines query intent.
    """
    ABBREVIATIONS = {
        "lang": "programming_language",
        "langs": "programming_language",
        "ide": "editor",
        "os": "environment operating system",
        "db": "database"
    }

    FILLER_PATTERNS = [
        r"\bcan you tell me\b",
        r"\bwhat is my\b",
        r"\bwhat are my\b",
        r"\bdo i have\b",
        r"\bplease show me\b"
    ]

    def rewrite(self, query: str) -> str:
        clean = query.lower().strip()
        for pattern in self.FILLER_PATTERNS:
            clean = re.sub(pattern, "", clean).strip()

        tokens = clean.split()
        expanded_tokens = [self.ABBREVIATIONS.get(t, t) for t in tokens]
        rewritten = " ".join(expanded_tokens).strip()

        return rewritten if rewritten else query
