# ChunkdUp — Independent Post-Implementation Technical Audit Report

**Audit Date:** 2026-10-14  
**Auditor Role:** Independent Senior AI Infrastructure Engineer  
**Audit Target Revision:** `44376c8bcfbf111aa29a6adbe6fa04d31cb4ac54`  
**Base Revision:** `54c1d108cb3299322f37cc0956acea3fd0eba691`  
**Verdict:** **`VERIFIED WITH LIMITATIONS`**

---

## 1. Verified Claims and Supporting Evidence

The table below catalogs claims made by the implementation agent alongside evidence verified independently during this audit.

| Claim | Status | Evidence / Verification Method |
| :--- | :---: | :--- |
| **All 8 architectural phases committed discretely** | **VERIFIED** | `git log -n 8` confirms 8 consecutive commits matching conventions (`2f92951` through `44376c8`). |
| **No secrets or credentials in commit history** | **VERIFIED** | Regex diff scan (`(?i)(api[_-]?key\|secret\|token\|password)`) over `54c1d10..HEAD` confirmed zero hardcoded secrets. Environment variables (`GROQ_API_KEY`, `LAYA_API_KEY`) are read via `os.getenv`. |
| **Frozen 92-query regression suite passes without regression** | **VERIFIED** | Ran `python experiments/retrieval/verify_production.py`. Output matched expected values: `without_demotion` MRR 0.8397, P@5 0.6366; `with_demotion` MRR 0.8424, P@5 0.6475. |
| **Raw dialogue turns survive IGNORE decisions** | **VERIFIED** | Tested via `test_raw_turn_saved_on_ignore` and `test_audit_lifecycle_trace_and_invariants`. Turn log preserves chit-chat while active memory store remains empty. |
| **Non-destructive temporal supersession preserves prior history** | **VERIFIED** | Tested via `test_fact_update_creates_supersession_chain` and `test_multi_version_supersession_lineage`. Updating a fact creates a new version, closes `valid_to`, and links `supersedes_id` / `superseded_by`. |
| **Point-in-time querying (`as_of`) functions accurately** | **VERIFIED** | Tested via `test_point_in_time_querying` and `test_memory_api_point_in_time_retrieval`. Queries reconstruct historical memory state before subsequent updates. |
| **Scale-aware calibrated abstention threshold $\tau(N)$** | **VERIFIED** | Tested via `test_grounding_gate.py`. As store size $N$ increases, $\tau(N)$ rises dynamically to prevent false retrieval spikes. |
| **Conservative entity linking with bounded 1-hop expansion** | **VERIFIED** | Tested via `test_entity_expansion.py`. Multi-hop relationships expand at depth=1 with distance decay penalties. |
| **Deterministic LoCoMo dev vs eval splits** | **VERIFIED** | Verified in `locomo_adapter.py`: `DEFAULT_DEV_SPLIT` (`conv-26`, `conv-30`) and `DEFAULT_EVAL_SPLIT` (`conv-42`..`conv-50`) are disjoint sets totaling 10 conversations. |
| **Reprocessing identical facts is idempotent** | **VERIFIED** | Tested via `test_audit_lifecycle_trace_and_invariants`. Re-stating an existing fact results in a `MERGE` decision, incrementing `frequency` rather than creating duplicate active records. |

---

## 2. Unverified Claims and Why They Remain Unverified

The following capabilities are implemented in code but could **not** be verified against live external services in this environment:

| Capability | Status | Reason Not Verified in this Audit |
| :--- | :---: | :--- |
| **Live Groq LLM Fact Extraction** | **UNVERIFIED (MOCKED ONLY)** | `GROQ_API_KEY` is not configured in the host environment (`GROQ_API_KEY is set: False`). Mocked tests pass (retry on 429, fallback on exception/malformed JSON), but live endpoint behavior was skipped (`test_live_groq_extraction`). |
| **Live PostgreSQL + pgvector Persistence** | **UNVERIFIED (SKIPPED)** | `DATABASE_URL` is not set. `tests/test_postgres.py` was skipped. The SQL schema and methods in `PostgresRepository` and `PostgresTurnRepository` exist, but no live PostgreSQL database was connected. |
| **Live Laya Decision Provider** | **SCAFFOLDED / UNVERIFIED** | There is no live Laya endpoint. The adapter (`LayaDecisionProvider`) is a thin subclass of `ExternalDecisionProvider` that does not even attach `Authorization: Bearer` headers. It is disabled by default (`CHUNKDUP_DECISION_PROVIDER=default`). |
| **End-to-End Retention on LoCoMo Track B** | **UNVERIFIED** | Benchmark metrics reported in `locomo_baseline.json` evaluate Track A (retrieval-only across ground-truth dialogue turns). Track B (end-to-end extraction and retention) was not run across the full dataset. |

---

## 3. Exact Test Commands and Observed Results

All commands were executed in the repository root (`D:\Development\Projects\core\ChunkdUp`) on Windows with Python 3.13.5:

### 3.1. Full Unit and Integration Test Suite
- **Command:** `$env:CHUNKDUP_CE_TIMEOUT_S="5.0"; pytest tests/`
- **Exit Code:** `0`
- **Observed Metrics:**
  - `collected 123 items`
  - `121 passed, 2 skipped, 2 warnings in 55.00s`
- **Skipped Tests Details:**
  1. `tests/test_groq_extraction.py::test_live_groq_extraction`: Skipped (`GROQ_API_KEY not set`).
  2. `tests/test_postgres.py::test_semantic_search`: Skipped (`DATABASE_URL not configured`).
- **Warnings:**
  - 2 deprecation warnings related to SWIG bindings in Python 3.13 (`DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute`).

### 3.2. Production Retrieval Verification
- **Command:** `$env:CHUNKDUP_CE_TIMEOUT_S="5.0"; python experiments/retrieval/verify_production.py`
- **Exit Code:** `0`
- **Observed Metrics:**
  - `without_demotion MRR 0.8397 (expected 0.8397) P@5 0.6366 (expected 0.6366) -> MATCH`
  - `with_demotion    MRR 0.8424 (expected 0.8424) P@5 0.6475 (expected 0.6475) -> MATCH`
  - `PRODUCTION VERIFICATION: PASS`

---

## 4. Benchmark Reproduction and Split Integrity

### 4.1. Frozen 92-Query Regression Suite
- The files `experiments/retrieval/dataset.jsonl` (70 queries) and `experiments/retrieval/dataset_round2.jsonl` (22 queries) were committed in commits `b8f8ef2` and `5b8ccc9` and have remained completely untouched.
- Fixture memories (`fixture.py`, 12 memories) remain identical.
- Target expectations (`with_demotion` MRR 0.8424, P@5 0.6475) reproduce exactly on the production retrieval pipeline.

### 4.2. LoCoMo Dataset & Split Integrity
- **Dataset:** `evaluation/data/locomo10.json` (10 conversations, 5,882 dialogue turns, 1,986 QA pairs).
- **Split Definitions:**
  - `dev` split: `["conv-26", "conv-30"]` (2 conversations).
  - `eval` split: `["conv-42", "conv-43", "conv-44", "conv-45", "conv-47", "conv-48", "conv-49", "conv-50"]` (8 conversations).
  - The splits are mutually exclusive and deterministically sorted by `sample_id`.
- **Evidence Parsing:**
  - `LoCoMoLoader.parse_evidence_ids` normalizes raw tokens (e.g., `D1:18` $\rightarrow$ `D1:18`).
  - Identified 1 malformed evidence token in `conv-42` question 88 (`"D"`), which is caught and explicitly reported in `validation_report["malformed_evidence_records"]`.
- **Metric Definitions:**
  - Full Recall@K ($1.0$ only if all ground truth turns are in top $K$) is explicitly distinguished from Partial Recall@K (at least 1 ground truth turn in top $K$) and Recall@K.
  - On the full dataset (`evaluation/results/locomo_baseline.json`), `Recall@5` is 0.6552, `PartialRecall@5` is 0.7279, and `FullRecall@5` is 0.5955.

---

## 5. Lifecycle and Provenance Findings

Tracing dialogue through the runtime reveals the following operational characteristics:

1. **Turn Persistence Before Processing:**
   In both `Memory.remember()` and `TurnBuffer.add_turn()`, dialogue turns are appended to `TurnRepository` **before** extraction rules or policy decisions are evaluated. Raw conversational dialogue is preserved even if the fact extractor is empty or policy returns `IGNORE`.
2. **Schema Invariant Defect Fixed During Audit:**
   `ExtractedFact` initially typed `source_turn_id: str` without checking for empty strings. An empty string `""` would satisfy basic Pydantic validation. During this audit, a `@field_validator("source_turn_id")` was added in `ai-system/chunkdup/extraction/schema.py` to guarantee non-empty provenance IDs on all facts.
3. **Idempotent Reprocessing:**
   When identical facts are ingested across sessions, `PolicyFactory` matches existing memory keys and types to issue `MERGE`, bumping `frequency` and recency without duplicating records.
4. **Temporal Invariants:**
   `UPDATE` actions preserve the superseded record by setting `status = "superseded"`, `valid_to = new_valid_from`, and `superseded_by = new_id`. Point-in-time querying via `as_of` filters records using `valid_from <= as_of < valid_to`.

---

## 6. Groq Integration Status

- **Status:** **MOCKED & UNIT-TESTED ONLY; LIVE INTEGRATION UNVERIFIED.**
- **Client Configuration:** Lazily initialized in `GroqFactExtractor` via `os.getenv("GROQ_API_KEY")`.
- **Fault Tolerance:**
  - 429 rate limit backoff: Tested with mocked client (`test_mocked_groq_retry_on_429`). Backoff uses exponential delay with full jitter.
  - Malformed LLM output: Tested with mocked client (`test_groq_malformed_json_fallback`). Falls back to pattern-based regex extractor.
  - Network / provider failure: Tested with mocked client (`test_groq_api_exception_fallback`). Falls back to pattern-based extractor without dropping dialogue turns.
- **Security:** No API keys are logged or printed.

---

## 7. Laya Integration Status

- **Status:** **SCAFFOLDED / UNVERIFIED (DISABLED BY DEFAULT).**
- **Findings:**
  - The `LayaDecisionProvider` is a stub subclassing `ExternalDecisionProvider`.
  - While it accepts `LAYA_ENDPOINT` and `LAYA_API_KEY`, the HTTP client in `ExternalDecisionProvider` does not set an `Authorization: Bearer <api_key>` header in requests.
  - No public documentation or verified remote endpoint exists for Laya.
  - The runtime defaults to `DefaultDecisionProvider` (`PolicyFactory`).
- **Recommendation:** Keep Laya marked as an unverified external integration. Do not enable in production.

---

## 8. Live PostgreSQL Test Status

- **Status:** **UNVERIFIED IN CURRENT ENVIRONMENT (CODE IMPLEMENTED).**
- **Findings:**
  - `PostgresRepository` and `PostgresTurnRepository` implement SQL schemas for append-only turns and temporal memory supersession.
  - Live tests in `tests/test_postgres.py` are skipped due to the absence of `DATABASE_URL`.
  - While unit tests pass for in-memory repositories, live PostgreSQL queries and pgvector index behavior cannot be claimed as verified until executed against an actual PostgreSQL 16+ instance.

---

## 9. Critical Defects Diagnosed and Minimal Fixes Applied

During the course of this independent audit, two defects were discovered and corrected with minimal safe edits:

1. **`evaluation/run_ablations.py` Metric Key Mismatch:**
   - *Defect:* `run_ablations.py` attempted to read `summary.get("overall")` and `summary.get("category_breakdown")` from `TrackAEvaluator`. However, `TrackAEvaluator._aggregate_metrics` returns `summary["overall_answerable"]` and `summary["categories"]`. This caused all overall LoCoMo metrics in `evaluation/results/ablations_report.json` to default to `0.0`.
   - *Fix:* Updated `evaluation/run_ablations.py` to extract `overall_answerable`, `categories`, and `overall_adversarial`.
2. **`ExtractedFact` Missing Non-Empty Validation on `source_turn_id`:**
   - *Defect:* Pydantic allowed empty strings `source_turn_id=""`, which could allow provenance-less facts to reach the memory manager if malformed input bypassed extraction.
   - *Fix:* Added `@field_validator("source_turn_id")` requiring non-empty, stripped strings in `ai-system/chunkdup/extraction/schema.py`.
3. **Comprehensive Lifecycle Audit Tests Added:**
   - Added `tests/test_audit_lifecycle.py` and additional mocked Groq tests in `tests/test_groq_extraction.py` covering malformed JSON and network exception recovery.

---

## 10. Tradeoff Analysis Across Evaluated Retrieval Configurations

| Configuration | MRR | P@5 | Latency (Steady State) | Latency (Cold Start) | Tradeoff Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `legacy_baseline` | 0.5185 | 0.4442 | < 1 ms | < 1 ms | **Fastest / Minimal Dependencies:** Zero ML models, pure Python string/BM25 matching. Lacks semantic generalization. |
| `dense_abstain_ed` | 0.8152 | **0.7005** | ~10–30 ms | ~45 s | **Best for Low-Latency Hot Paths:** Dense embeddings with cosine abstention gate and entity demotion. Highest precision (70.05%), fast steady state, but first query suffers model initialization latency. |
| `hybrid_ce_abstain` | **0.8424** | 0.6475 | ~650–750 ms | ~65 s | **Highest Ranking Fidelity:** Hybrid dense+BM25 with cross-encoder reranking. Best MRR (84.24%), but 20x higher latency on CPU. |

**Audit Conclusion on Tradeoffs:**
There is no single "best" configuration. `dense_abstain_ed` is optimal for real-time per-turn agent interactions where latency budgets are under 50ms. `hybrid_ce_abstain` is optimal for deeper deliberative reasoning where ranking fidelity warrants 700ms. Latency measurement harnesses should include a warm-up phase to avoid skewing averages with one-time model weight loading.

---

## 11. Final Verdict

**VERDICT: `VERIFIED WITH LIMITATIONS`**

- **Verified:** Packaging, zero regression on frozen 92 queries, in-memory repository, temporal append-only data model, point-in-time querying, evidence-grounded retrieval, calibrated abstention, conservative entity graph, and fallback exception handling.
- **Limitations:** Live Groq extraction, live PostgreSQL persistence, and Laya decision provider are unverified in this environment due to missing live external credentials/endpoints.
