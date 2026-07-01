# ChunkdUp: My LLM Context-Engineering Learning Journey

Welcome to my personal playground for learning, building, and optimizing **Context Engineering** systems for Large Language Models (LLMs). As I implement each concept step-by-step, I document my progress, experiments, results, and engineering decisions.

---

## Repository Structure

- `ai-system/labs/` — Active experiments, implementation code, and learning logs.
  - `001-context-assembly/` — Lab focused on retrieval methods, budget management, and prompt construction.
    - `experiments/` — Python scripts executing retrieval and LLM call flows.
    - `learnings/` — Distilled reflections and key insights from each experiment.
    - `data/` — Local chunks and mock documents.

---

##  Completed Labs
### Lab 001: Context Assembly
**Core Question:** *Given retrieved data chunks, how should an AI system decide what reaches the LLM?*

#### 1. Keyword Retrieval & Top-3 Assembly
- **What We Built:**
  - A basic keyword-overlap matching retriever (`_calculate_overlap`).
  - A naive `ContextAssembler` that simply takes the **Top-3** highest-ranking chunks.
- **What We Learned:**
  - **Keyword matching is brittle:** Counting exact word matches is highly sensitive to punctuation, stop words, and vocabulary mismatch.
  - **Fixed Top-3 is size-blind:** Selecting a static number of chunks ignores the fact that different chunks have different lengths, risking context window overflow or under-utilization.

#### 2. Semantic Retrieval, Budget-Aware Packing, & Structured Outputs
- **What We Built:**
  - **Semantic Retriever:** Upgraded keyword search to dense vector search using `SentenceTransformer` (`all-MiniLM-L6-v2`) and cosine similarity.
  - **Budget-Aware Assembler:** Designed a dynamic chunk packing algorithm that fills a defined character budget (e.g., 800 chars) greedily without exceeding it.
  - **Prompt Orchestration:** Built a system to configure basic/expert prompt variants and format context chunks with source numbers.
  - **JSON Output Constraints:** Instructed the LLM to output a machine-consumable JSON schema containing the answer, a confidence score, and chunk citations.
  - **Multi-Provider LLM Wrapper:** Unified API interactions for Google Gemini (`gemini-2.5-flash`) and Anthropic Claude (`claude-3-5-haiku-20241022`).
- **What We Learned:**
  - **Semantic matching is superior:** Embedding-based retrieval captures meaning rather than just matching characters.
  - **Greedy budget-packing maximizes utility:** By prioritizing by score and checking size constraints before adding, we prevent truncation while packing as much relevant info as possible.
  - **Structured LLM outputs are crucial:** Forcing the model to return JSON with `"citations"` (e.g., `[1, 6]`) and `"confidence"` makes it easy to integrate LLM answers programmatically and verify source ground truth.

---

##  My Engineering Decisions

* **Semantic > Keywords:** Adopted semantic retrieval as the default retrieval strategy.
* **Budget-Aware > Fixed Top-K:** Budget-aware selection is the standard, ensuring robust handling of LLM context window limits.
* **Non-Blocking Oversized Chunks:** If a highly relevant chunk is too large to fit in the remaining budget, we skip it and try to pack subsequent smaller (but still relevant) chunks instead of stopping assembly immediately.
* **Decoupled Orchestration:** Kept retriever, assembler, and prompt builder classes separated so they can be iterated on independently.
* **Structured Output Design:** Prompt instructions should require JSON payloads to allow deterministic downstream parsing.

---

##  Open Questions for Next Labs

- **When should reranking happen?** (Should we retrieve a large pool `K=50`, rerank them, and then assemble the top ones?)
- **Should we filter out low-score chunks?** (Avoiding low-quality "noise" even if we have remaining budget).
- **How should context compression fit into the pipeline?** (Using summaries or LLM-based filtering before assembly).

---


