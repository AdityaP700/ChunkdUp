# ChunkdUp Research Plan: Evidence-Preserving Associative Memory for AI Agents

**Date**: 2026-10-17  
**Branch**: `main`  
**Base Commit**: `3d44178` (`feat(retrieval): optimize graph expansion, two-tier profile memory, and query routing across LoCoMo dev and held-out benchmarks`)  
**Status**: Approved for Implementation

---

## 1. Executive Summary & Objective

The objective of this research sprint is to investigate whether a focused, associative memory architecture improves conversational retrieval in ChunkdUp by combining:
1. **Original Conversational Experiences** (authoritative turn-level ground truth).
2. **Grep-style Exact Lexical Retrieval and BM25** (transparent, uncorrupted keyword/phrase matching).
3. **Explicit Relationships between Entities, Events, and Facts** (associative graph edges).
4. **Selective Semantic Consolidation** (atomic facts retaining bidirectional source provenance).
5. **Query-Driven Retrieval and Evidence Selection** (intent-aware dynamic routing).

The benchmark is the official **LoCoMo** evaluation dataset. LongMemEval is explicitly out of scope for this sprint. Every added component must earn its place through controlled evaluation against an unchanged baseline.

---

## 2. Research Hypotheses

All hypotheses are treated as strictly falsifiable:

- **H1 (Bounded Graph Expansion)**: Bounded graph expansion applied after strong initial retrieval will improve evidence retrieval on questions requiring multiple related memories (specifically LoCoMo Category 3 multi-hop questions), compared with the existing hybrid retrieval baseline.
- **H2 (Semantic Consolidation with Provenance)**: Semantic consolidation into atomic facts can improve retrieval or reduce the context required to answer questions, provided that the original turns remain available and each derived fact retains source provenance.
- **H3 (Multi-Representation Synergy)**: A hybrid system selecting evidence across raw turns, semantic facts, and graph-linked memories will outperform a single-representation system on a balanced set of conversational questions.
- **H4 (Domain & Query Heterogeneity)**: These benefits are query-dependent. Graph expansion may introduce noise on simple single-hop queries, consolidation may discard nuanced details, and additional graph/reranking operations increase latency. We will systematically isolate where each approach helps and where it hurts.

---

## 3. Minimal Memory Representation (Phase 2)

ChunkdUp avoids bloated cognitive frameworks. The data model is minimal, lean, and standard-library-centric:

### A. Experience Memory
- **Purpose**: Authoritative evidence ground truth.
- **Fields**: `dia_id` (e.g. `D1:3`), `speaker`, `text`, `session_num`, `timestamp` (ISO datetime or formatted session string), `contextual_text`.
- **Invariance**: Raw turns are never mutated or silently overwritten when facts change.

### B. Semantic Memory
- **Purpose**: Atomic extracted propositions for granular semantic matching and context reduction.
- **Fields**: `subject`, `predicate`, `object`, `source_turn_ids` (list of `dia_id`s), `timestamp`, `event_time`, `confidence`, `validity_interval`.
- **Provenance Requirement**: Every semantic fact must point back to the exact turn IDs that support it.

### C. Associative Memory
- **Purpose**: Typed relational links connecting entities, events, and facts.
- **Relationship Vocabulary**:
  - `entity_to_event`: connects an entity to an event in which it participated.
  - `event_to_event`: temporal or causal succession between dialogue events.
  - `fact_to_source`: explicit pointer linking an extracted fact to its grounding turn.
  - `state_transition`: connects an earlier state or preference to an updated/superseded state.
  - `entity_relation`: subject-predicate-object semantic relation.
- **Co-occurrence Guard**: Mere co-occurrence in a sliding window is not treated as causation.

### D. Optional Consolidated Profile
- **Purpose**: High-level key-value user/assistant attributes.
- **Governance**: Kept strictly as an auxiliary view, not a canonical evidence store. Always points to underlying evidence turns.

---

## 4. Experimental Matrix: 7 Retrieval Variants (Phase 3)

| Variant | Name | Representation | Traversal / Rerank Mechanics |
| :--- | :--- | :--- | :--- |
| **A** | `GrepRetriever` | Raw turns | Normalized case-insensitive substring & token phrase matching |
| **B** | `BM25Retriever` | Raw turns | Pure BM25Okapi scoring with standard stopword filtering |
| **C** | `HybridCEControl` | Raw turns + memories | Dense (`all-MiniLM-L6-v2`) + BM25 + Cross-Encoder (`ms-marco-MiniLM-L-12-v2`) + Abstention |
| **D** | `BoundedAssociative` | Raw turns + Associative links | Hybrid seeds $\rightarrow$ Bounded graph expansion ($D \le 2, B \le 3$, cap 5) $\rightarrow$ CE rerank |
| **E** | `ConsolidatedSemantic` | Raw turns + Semantic facts | Joint index; retrieved facts expand to underlying `source_turn_ids` |
| **F** | `ConsolidatedAssociative` | Raw turns + Facts + Links | Joint semantic facts and bounded associative graph links |
| **G** | `QueryRoutedAssociative` | Dynamic multi-representation | Deterministic query classifier routing to optimal strategy per category |

---

## 5. Metrics & Evaluation Protocol (Phases 1, 4, 5)

### Evaluation Splits
- **Development Split**: `conv-26` and `conv-30` (788 dialogue turns across multiple sessions, 304 annotated questions).
- **Held-Out Split**: `conv-41` through `conv-50` (5,094 dialogue turns, 1,682 annotated questions).

### Metrics
1. **Hit@k**: Top-$k$ hit rate ($k \in \{1, 3, 5, 8\}$) against gold evidence turn IDs.
2. **MRR**: Mean Reciprocal Rank of the first gold evidence turn.
3. **Evidence Recall@k**: Fraction of all annotated gold turns for a question retrieved in top-$k$.
4. **Multi-Hop Full Recall**: Percentage of multi-hop questions (Category 3) where **all** required gold turns are retrieved in top-$k$.
5. **Temporal Accuracy**: Hit rate and evidence accuracy on Category 2 (temporal reasoning) questions.
6. **Abstention Metrics**: Abstention rate on unanswerable/adversarial questions (Category 5), false abstention rate on answerable questions.
7. **Efficiency**: Retrieval latency (p50 and p95 in ms, warm steady-state on RTX 4050 GPU) and average retrieved context token length.

### Failure Taxonomy (Categories A through J)
- **A**: Extraction failure (missing/incorrect fact extraction).
- **B**: Lexical miss (exact terms missing or vocabulary mismatch).
- **C**: Semantic miss (embedding distance too large).
- **D**: Incorrect graph link (spurious relational edge).
- **E**: Graph expansion noise (irrelevant neighbors dilute ranking).
- **F**: Missing multi-hop evidence (only 1 of $N$ hops found).
- **G**: Temporal conflict / stale state retrieved.
- **H**: Consolidation / compression loss (fine details omitted from fact).
- **I**: Ranking failure (evidence present in candidates but pushed past top-$k$).
- **J**: Incorrect abstention (valid answerable question suppressed).

---

## 6. Execution Roadmap & Milestones

1. **Phase 0**: Plan artifact and repository plan deliverable (`docs/research/associative-memory-plan.md`). Initialize `CHANGELOG.md`. Commit to Git.
2. **Phase 1**: Run & verify clean baseline commands for Track A and Track B.
3. **Phase 2 & 3**: Implement schema, graph, and the 7 retrieval variants in `ai-system/chunkdup/associative/`.
4. **Phase 4**: Execute development split comparative benchmark across all 7 variants; run controlled ablations.
5. **Phase 5**: Compute full metrics, latency profiles, context lengths, and failure taxonomy. Run held-out split validation.
6. **Phase 6**: Unit & integration tests in `tests/test_associative_memory.py`. Verify production compatibility and 100% test pass.
7. **Phase 7 & 8**: Produce final comprehensive research report (`docs/research/associative-memory-report.md`). Final review and push.
