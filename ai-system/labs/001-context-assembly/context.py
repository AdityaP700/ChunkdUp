class ContextAssembler:
    def __init__(self):
        pass

    def assemble(self, chunks: list[dict], policy: str = "top_3", max_chars: int = 150, threshold: float = 0.53) -> list[dict]:
        """
        Assembles chunks into a context based on the chosen policy and constraints.
        
        Args:
            chunks: List of retrieved chunks, each containing 'text' and 'score'.
            policy: One of 'top_3', 'budget_aware', 'threshold_budget'.
            max_chars: Character budget limit for the assembled context.
            threshold: Minimum similarity score threshold for threshold_budget policy.
        """
        if not chunks:
            return []

        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get("score", 0.0),
            reverse=True
        )

        if policy == "top_3":
            return sorted_chunks[:3]

        elif policy == "budget_aware":
            # Budget-Aware: add chunks in sorted order until budget limit is reached
            selected = []
            current_chars = 0
            for chunk in sorted_chunks:
                chunk_len = len(chunk["text"])
                if current_chars + chunk_len <= max_chars:
                    selected.append(chunk)
                    current_chars += chunk_len
            return selected

        elif policy == "threshold_budget":
            # New Policy: Only consider chunks above the threshold, then apply budget constraint
            selected = []
            current_chars = 0
            for chunk in sorted_chunks:
                if chunk.get("score", 0.0) < threshold:
                    continue
                chunk_len = len(chunk["text"])
                if current_chars + chunk_len <= max_chars:
                    selected.append(chunk)
                    current_chars += chunk_len
            return selected

        else:
            raise ValueError(f"Unknown policy: {policy}")
