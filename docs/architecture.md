# ChunkdUp Architecture & Production Readiness Reference

This document provides a comprehensive technical overview of the ChunkdUp persistent memory runtime, detailing its layered ingestion architecture, temporal append-only data model, evidence-grounded retrieval pipeline, conservative entity graph, decision provider interface, and empirical benchmark results.

---

## 1. High-Level System Architecture

ChunkdUp is designed as an evidence-grounded persistent memory runtime for AI agents. Rather than treating conversational memory as an unstructured dump of vector embeddings, ChunkdUp enforces strict separation of concerns across raw conversation preservation, fact extraction, lifecycle policy decisions, temporal indexing, and multi-stage grounded retrieval.

```
Conversational Dialogue Stream
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 0: Raw-First Append-Only Turn Log                     │
│ (TurnRepository: InMemoryTurnRepository / PostgresTurnRepo) │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 1: Heuristic Eligibility Filter                       │
│ (is_turn_eligible: filters greetings, filler, acknowledg.)   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 2: Batched Extraction with Rate-Limit Backoff         │
│ (TurnBuffer -> GroqFactExtractor with 429 jitter backoff)   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 3: Strict Schema Validation & Normalization           │
│ (Pydantic ExtractedFact: fact, category, confidence, etc.)  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 4: Decision Engine & Pluggable Decision Provider      │
│ (DecisionEngine -> STORE / MERGE / UPDATE / IGNORE)         │
│ (DefaultDecisionProvider, ExternalDecisionProvider, Laya)   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Memory Persistence & Temporal Append-Only Ledger            │
│ (valid_from, valid_to, supersedes_id, superseded_by)        │
│ EntityGraph Indexing (aliases, co-occurrence, 1-hop edges)   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Grounded Retrieval & Calibrated Abstention Gate             │
│ (Dense + BM25 -> RRF -> Cross-Encoder -> Entity Demotion    │
│  -> Calibrated Abstention tau(N) -> Speaker/Contradiction)  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Ingestion Pipeline & Fact Extraction

Memory formation operates across five strict architectural levels:

### Level 0: Raw-First Append-Only Turn Log
All conversational dialogue turns ingested via `memory.remember(..., record_turn=True)` or `memory.record_turn(speaker, text)` are immediately and permanently recorded into a `TurnRepository` (`SourceTurnLog`).
- **Invariant**: Raw dialogue is **never deleted or dropped**, even if subsequent extraction or policy stages yield an `IGNORE` decision.
- Every turn is assigned a unique `turn_id`, monotonically increasing `turn_number`, timestamp, and speaker identity.

### Level 1: Turn Eligibility Filtering
Before invoking external LLMs, the incoming turn passes through `is_turn_eligible(turn)`. Turns consisting solely of greetings ("hello", "thanks"), acknowledgments ("ok", "got it"), or trivial system chatter are filtered out to prevent unnecessary API latency and token expenditure.

### Level 2: Batched Extraction with Rate-Limit Resilience
Eligible turns are buffered in a `TurnBuffer`. When flushed (via batch size or timeout), facts are extracted using `GroqFactExtractor`:
- Utilizes fast inference models (e.g., `llama-3.1-8b-instant`).
- Robust handling of HTTP 429 / rate limits via exponential backoff with full jitter.
- Schema recovery: Structured JSON output with a regex fallback parser to guarantee zero dropped payloads under malformed LLM responses.

### Level 3: Pydantic Schema Validation
Extracted facts must conform to the strict `ExtractedFact` Pydantic model:
- `fact` (str): Precise proposition stated in the conversation.
- `category` (FactCategory): `preference`, `biographical`, `project`, `belief`, `technical`, `general`.
- `confidence` (float): Extraction certainty in $[0.0, 1.0]$.
- `validity` (FactValidity): `active`, `provisional`, `superseded`.
- `source_turn_id` (str): Explicit backlink to the origin turn in the Level 0 log.
- `subject` (Optional[str]) and `entity_names` (List[str]): Named entities for graph indexing.

### Level 4: Decision Engine & Decision Providers
Candidate facts are compared against existing active memories in `DecisionEngine` to produce one of four lifecycle actions:
1. `STORE`: Novel fact added to memory.
2. `MERGE`: Existing fact corroborated, incrementing frequency and bumping confidence/recency.
3. `UPDATE`: Existing fact modified or superseded by a newer contradictory/refined fact.
4. `IGNORE`: Redundant or low-salience fact dismissed from active memory (while remaining safely in the Level 0 raw turn log).

Decision logic can be customized via the pluggable `DecisionProvider` protocol (`DefaultDecisionProvider`, `ExternalDecisionProvider`, or `LayaDecisionProvider`).

---

## 3. Temporal Append-Only Memory Model

ChunkdUp avoids destructive updates. In-place overwrites are prohibited across both `InMemoryRepository` and `PostgresRepository`.

### Schema & Metadata Fields
Every memory record maintains temporal tracking:
- `created_at` (datetime): Timestamp when the memory was first recorded.
- `updated_at` (datetime): Timestamp of latest corroboration or merge.
- `valid_from` (datetime): Point in time from which the fact is considered true.
- `valid_to` (Optional[datetime]): Point in time at which the fact was superseded (`None` = currently active).
- `supersedes_id` (Optional[str]): Memory ID of previous state this fact replaces.
- `superseded_by` (Optional[str]): Memory ID of newer state that replaced this fact.

### Supersession Lifecycle
When an `UPDATE` decision occurs:
1. The prior active memory record is closed by setting `valid_to = now()` and `superseded_by = new_memory_id`.
2. The new memory record is inserted with `valid_from = now()`, `valid_to = None`, and `supersedes_id = prior_memory_id`.
3. History is fully preserved.

### Point-in-Time Querying (`as_of`)
Both in-memory and PostgreSQL repositories support temporal point-in-time querying via `as_of=datetime`:
- Only memories where `valid_from <= as_of` and (`valid_to is None` or `valid_to > as_of`) are returned.
- Allows agents to reconstruct past state and evaluate counterfactuals accurately.

---

## 4. Grounded Retrieval & Calibrated Abstention

Retrieval in ChunkdUp is designed to prioritize precision, evidence grounding, and deliberate abstention over hallucinations.

### Multi-Stage Retrieval Pipeline
1. **Hybrid Retrieval**:
   - Dense semantic retrieval via `sentence-transformers/all-MiniLM-L6-v2`.
   - Sparse keyword retrieval via `rank_bm25.BM25Okapi`.
   - Candidates fused using Reciprocal Rank Fusion (RRF, $k=60$).
2. **Cross-Encoder Reranking**:
   - Top candidates reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
3. **Type-Aware Entity Demotion**:
   - Penalizes memories whose detected entity types contradict the query's target entity types.
4. **Calibrated Abstention Gate ($\tau(N)$)**:
   - Scale-aware dynamic threshold:
     $$\tau(N) = \tau_0 + \alpha \ln(1 + \beta N)$$
   - Prevents false retrieval spikes as the memory store grows.
5. **Evidence Grounding Verification (`GroundingGate`)**:
   - Validates candidate memories against query intent, speaker alignment, and contradiction flags.
   - If grounding fails or top score is below $\tau(N)$, returns an empty result with `abstained=True` and a clear abstention reason.

---

## 5. Conservative Entity Linking & Bounded 1-Hop Expansion

To handle multi-hop and relational questions without unconstrained graph explosion, ChunkdUp incorporates `EntityGraph`:

- **Entity & Alias Index**: Maintains canonical entity representations, resolving aliases and case variations to unique entity IDs.
- **Co-occurrence Edges**: Captures entities co-occurring within the same memory or conversational turn.
- **Bounded 1-Hop Expansion**:
  - When a query matches specific entities, the graph retrieves 1-hop neighbor memories.
  - Expansion is strictly bounded by max candidate limits and depth=1.
  - Scores of expanded memories receive distance decay penalties to prevent drift.

---

## 6. Empirical Benchmark Results & Verification

ChunkdUp's retrieval pipeline is validated against frozen benchmarks to ensure zero regression.

### Frozen 92-Query Regression Suite
Verified using `experiments/retrieval/verify_production.py`:
- **Without Demotion Baseline**:
  - MRR: **0.8397** (Target: $\ge 0.8397$) — **MATCH**
  - Precision@5: **0.6366** (Target: $\ge 0.6366$) — **MATCH**
- **With Demotion (Production Standard)**:
  - MRR: **0.8424** (Target: $\ge 0.8424$) — **MATCH**
  - Precision@5: **0.6475** (Target: $\ge 0.6475$) — **MATCH**

### Configuration Ablation Scoreboard (`evaluation/results/ablations_report.json`)
Evaluated across frozen 92 queries:
| Configuration | MRR | Precision@5 | Avg Latency (ms) | Description |
| :--- | :---: | :---: | :---: | :--- |
| `legacy_baseline` | 0.5185 | 0.4442 | 0.6 ms | BM25 + heuristic ranking (zero ML models) |
| `dense_abstain_ed` | 0.8152 | 0.7005 | 489.9 ms | Fast neural path: dense + abstention + entity fix |
| `hybrid_ce_abstain` | **0.8424** | **0.6475** | 667.9 ms | Production: hybrid + cross-encoder + entity demotion + gate |

---

## 7. Production Readiness & Configuration Reference

### Environment Variables
- `CHUNKDUP_RETRIEVAL_CONFIG`: Rollback/override configuration (`hybrid_ce_abstain`, `dense_abstain_ed`, `baseline_current`).
- `CHUNKDUP_CE_TIMEOUT_S`: Maximum allowed cross-encoder rerank latency in seconds (default: `1.0`). If exceeded, gracefully degrades to `dense_abstain`.
- `CHUNKDUP_QUERY_LOG`: Query audit log toggle (`on` / `off`).
- `GROQ_API_KEY`: API key for Level 2 LLM extraction (optional; fallback to heuristic/direct memories if unset).

### SDK Public API Example
```python
from chunkdup import Memory

# 1. Initialize Memory with in-memory store
memory = Memory(store="memory")

# 2. Record raw conversational dialogue
turn_id = memory.record_turn(speaker="User", text="I use FastAPI for web microservices.")

# 3. Form memory state
decision = memory.remember("I use FastAPI for microservices.")
print(decision)  # {"action": "STORE", ...}

# 4. Point-in-time querying
active_memories = memory.get_all_memories()

# 5. Evidence-grounded retrieval
results = memory.retrieve("What web framework do I use?", k=3)
for res in results:
    print(res["content"], res["score"])
```
