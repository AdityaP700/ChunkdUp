# Lab 003: Building a Resilient Memory Pipeline

## Core Objective
Evolving our AI system's memory from simple key-value pairs into a robust, policy-driven pipeline that correctly extracts, evaluates, and persists user context over time.

## 1. Evolution of Extraction
We transitioned from a rigid, hardcoded rule-based parser to a declarative **MemoryExtractor**. By utilizing Regex rules mapped to specific memory properties (type, key, value_group, meta), we successfully structured the ingestion layer. Future iterations can cleanly swap Regex for NLP (spaCy) or LLM-based structured extraction without breaking downstream components.

## 2. Rich Metadata Schema
Instead of raw key-value pairs, memories are now robust entities enriched with metadata:
- `id`: Unique identifier (UUID)
- `type`: Categorization (e.g., project, environment, preference)
- `created_at` / `updated_at`: Temporal tracking
- `source` & `status`: Provenance and lifecycle management
- `confidence`: Extraction certainty
- `meta`: Type-specific properties (e.g., `language` for projects)

## 3. Orchestrating the Memory Lifecycle
We established a clean dependency injection pipeline separating concerns:
```text
Conversation
      │
MemoryExtractor (Parses text into Candidate Memories)
      │
MemoryManager (Orchestrates the workflow)
      │
DecisionEngine (Evaluates conflict resolution via Policies)
      │
MemoryRepository (Handles CRUD operations to memory.json)
```
The **MemoryManager** handles the core orchestration: it fetches the existing active memory from the repository by key, consults the DecisionEngine, and executes the resulting decision (`STORE`, `IGNORE`, or `UPDATE`).

## 4. Policy-Driven Decision Engine
**The Problem:** Not all memories evolve the same way. A change in a project's language requires an `UPDATE` even if the project name remains identical, while an environment preference only updates if the primary value changes.

**The Solution:** We implemented the **Strategy Pattern** to handle type-specific behaviors. 
- We built a `PolicyFactory` that routes candidate memories to specific policies (`EnvironmentPolicy`, `ProjectPolicy`, `ToolPolicy`, `PreferencePolicy`).
- The `DecisionEngine` is now a thin router: `policy = PolicyFactory.get(memory["type"])`.
- The complexity of *how* a memory evolves is encapsulated inside its respective policy class.

## 5. Dynamic Persistence
The **MemoryRepository** was upgraded to handle targeted updates. When a policy dictates an `UPDATE`, the repository finds the matching active memory and selectively merges the new fields (e.g., updating the `language` and bumping the `updated_at` timestamp) while preserving the original `id` and `created_at` data.

## Next Steps
With extraction, evaluation, and persistence cleanly handled, the next logical step is integrating this persistent memory store into the context assembly pipeline, allowing the AI to dynamically recall this state during runtime prompt synthesis.