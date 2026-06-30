# Engineering Question

How should an AI system decide what information reaches the model?

# Experiments

- Keyword Retrieval
- Semantic Retrieval
- Top-K Assembly
- Budget-Aware Assembly
- Prompt Variants
- Structured Output

# Decisions

- Semantic retrieval replaced keyword matching.
- Budget-aware assembly is preferable to fixed Top-K.
- Oversized chunks should be skipped rather than stopping assembly.
- Prompt formatting is a separate concern from retrieval.
- Structured JSON outputs make LLM responses machine-consumable.

# Open Questions

- When should reranking happen?
- Should low-score chunks be filtered?
- How should compression fit into the pipeline?