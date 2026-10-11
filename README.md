# ChunkdUp

> **An experimental memory runtime for AI agents , built from scratch to understand what it actually means for a machine to remember.**

Yes, I named it ChunkdUp purely because it sounded cool in my head. There is no deeper meaning. Don't look for one.

I'm building this because I don't want to treat "AI memory" as a magical API call.

When we say an agent "remembers" something, several different problems are hiding underneath:

- What information is actually worth remembering?
- What should happen when the same information appears again?
- What happens when a new fact contradicts an old one?
- How do we persist memory without turning the database into a dump of everything the user ever said?
- How do we retrieve the right memory when the user's query is vague, incomplete, or indirect?
- How do we know whether the memory system is actually working?

ChunkdUp is my attempt to answer these questions by building the underlying pieces myself, testing them, and keeping track of where the system still fails.

---

## V1: What We Have Built

V1 focuses primarily on **memory formation, state management, persistence, and establishing a measurable retrieval baseline**.

The core idea is simple:

```text
Conversation
     │
     ▼
Extract useful information
     │
     ▼
Memory Decision
     │
     ├── STORE
     ├── MERGE
     ├── UPDATE
     └── IGNORE
     │
     ▼
Persistent Memory
     │
     ▼
Retrieval
     │
     ▼
Relevant Context
````

Rather than treating every conversation as something that should automatically become memory, ChunkdUp makes the memory lifecycle explicit.

### Example

```text
"I use Python."
        ↓
      STORE

"I use Python."
        ↓
      MERGE

"I switched my backend to Go."
        ↓
      UPDATE

"It's raining today."
        ↓
      IGNORE
```

The result is not just a collection of vectors.

It is a piece of state that can evolve over time.

---

# The Journey So Far : The Messy Labs

ChunkdUp started as a collection of small experiments. Each lab explored a problem that eventually became part of the larger system.

## Lab 001 : The Constraint Problem

**How do we decide what information survives when context is limited?**

[Read Lab 001](learnings/lab001/leanings_001.md)

Explored:

* context constraints
* keyword-based selection
* semantic similarity
* greedy budget packing

The first lesson was simple:

> Matching words isn't the same as matching meaning.

---

## Lab 002 : The Trust Problem

**How do we safely consume output from a probabilistic model?**

[Read Lab 002](learnings/lab002/learnings_002.md)

Explored:

* structured JSON output
* schema validation
* regex fallback parsing
* intentional rejection
* handling malformed model output

The goal was not to make the LLM magically reliable.

It was to make the system **safe when the LLM isn't reliable**.

---

## Lab 003 : The State Problem

**How should memory evolve when new information arrives?**

[Read Lab 003](learnings/lab003/lab3_complete_overview.md)

This became the core of ChunkdUp's memory lifecycle.

The system explicitly handles:

```text
STORE
MERGE
UPDATE
IGNORE
```

and maintains memory state rather than blindly accumulating new records.

This is also where the project moved from being a collection of experiments toward becoming an actual memory runtime.

---

## Lab 004 : The Semantic Memory Pipeline

**How do we turn stored memory into useful context for an AI system?**

[Read Lab 004](learnings/lab004/learnings_004.md)

This connected the pieces:

```text
User Query
     ↓
Memory Retrieval
     ↓
Context Assembly
     ↓
LLM
     ↓
Structured Output
```

The system could now retrieve dynamic user state and use it as part of an AI interaction.

---

# V1 Architecture

The current V1 architecture looks like this:

```mermaid
flowchart TD

    UserQuery[User Query / Conversation]

    subgraph MemoryFormation[Memory Formation]
        Extractor[Memory Extractor]
        Decision[Decision Engine]
        Repo[(Memory Repository)]

        Extractor -->|Extract facts| Decision
        Decision -->|STORE / MERGE / UPDATE / IGNORE| Repo
    end

    UserQuery --> Extractor
    UserQuery --> Retriever

    subgraph Retrieval[Memory Retrieval]
        Retriever[Hybrid Retriever]
        Semantic[Semantic Retrieval]
        Keyword[Keyword Retrieval]
        Ranking[Ranking / Reranking]

        Retriever --> Semantic
        Retriever --> Keyword
        Semantic --> Ranking
        Keyword --> Ranking
    end

    Repo -->|Active Memories| Retriever

    Ranking --> PromptBuilder[Context Assembly]

    subgraph Generation[LLM Interaction]
        PromptBuilder --> LLM[LLM]
        LLM --> Parser[Output Parser]
        Parser --> Validator[Output Validator]
        Validator --> FinalOutput([Structured Output])
    end
```

## What's Next?

This chapter of ChunkdUp is complete! We have successfully built a modular, deterministic, memory-aware infrastructure pipeline from the ground up without relying on black-box frameworks.

We will be moving forward toward new horizons and different learnings. Keep building!

## Retrieval configuration & incident rollback

ChunkdUp's default retrieval pipeline is `hybrid_ce_abstain` (dense + BM25 → RRF → cross-encoder rerank → calibrated abstention gate; see `docs/round3-decision.md` for the evidence behind this choice). If the pipeline misbehaves in production — bad answers, latency spikes, model problems — you can force the frozen, pre-experiment baseline (keyword + entity + heuristic reranking, no ML models) for every `Memory` instance without touching code, by setting one environment variable before the process starts:

```bash
CHUNKDUP_RETRIEVAL_CONFIG=baseline_current   # force the frozen round-1 baseline
CHUNKDUP_RETRIEVAL_CONFIG=dense_abstain      # or the fast path (~10ms, no cross-encoder)
```

This exists because a retrieval regression is best handled by falling back to the last known-good configuration in one move, not by debugging in production; the flag is read at retriever construction, so a restart applies it. Additionally, the pipeline degrades automatically and logs every degradation: if the cross-encoder fails to load, errors, or exceeds its timeout (`CHUNKDUP_CE_TIMEOUT_S`, default 1.0s), retrieval continues on the dense gate; if embeddings are unavailable it continues on the baseline path. Every query is appended to `<data_dir>/retrieval_log.jsonl` (disable with `CHUNKDUP_QUERY_LOG=off`) for post-hoc inspection.
