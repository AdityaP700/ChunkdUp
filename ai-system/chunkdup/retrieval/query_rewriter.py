# ai-system/chunkdup/retrieval/query_rewriter.py
import re

class QueryRewriter:
    """
    Cleans filler words, expands domain abbreviations and synthesis queries,
    while preserving vital semantic context for preferences and career lookups.
    """
    ABBREVIATIONS = {
        "lang": "programming_language language",
        "langs": "programming_language language",
        "ide": "editor ide",
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

        # Fix 5: Synthesis query expansion
        if any(w in clean for w in ["stack", "tech stack", "tools", "setup", "environment", "overview"]):
            return "programming_language project_name editor ide database environment cloud_provider tech stack tool"

        # Preference query expansion
        if any(w in clean for w in ["respond", "response", "style", "communication", "how should"]):
            return "preference response_style concise technical respond style"

        # Fix 3: Preserve preference and career query context (don't strip filler if query is preference/career)
        is_preference_or_career = any(w in clean for w in ["respond", "style", "work", "role", "job", "company", "employer"])
        if not is_preference_or_career:
            for pattern in self.FILLER_PATTERNS:
                clean = re.sub(pattern, "", clean).strip()

        tokens = clean.split()
        expanded_tokens = [self.ABBREVIATIONS.get(t, t) for t in tokens]
        rewritten = " ".join(expanded_tokens).strip()

        return rewritten if rewritten else query
