# Lab 003: The State Problem (Resilient Memory Pipeline)

## Core Objective
Evolving our AI system's memory from dumb key-value pairs into a resilient, policy-driven pipeline that correctly extracts, evaluates, and persists user context over time. Because apparently, just appending everything to a JSON file until it crashes isn't good software engineering.

## 1. Evolution of Extraction
We threw away the rigid, hardcoded string parsing logic and built a declarative **MemoryExtractor**. By mapping Regex rules to specific memory properties, we built a somewhat intelligent ingestion layer. One day we'll swap the Regex for an LLM extractor, but for now, it gets the job done without breaking downstream components.

## 2. Rich Metadata Schema
Instead of raw key-value pairs, memories are now robust entities enriched with metadata:
- `id`: So we can actually track things.
- `type`: Categorization (because a project is not the same as an operating system).
- `created_at` / `updated_at`: Because time is linear.
- `source` & `status`: Provenance tracking.
- `confidence`: How sure are we that the user actually meant this?
- `meta`: Extra type-specific garbage we want to save (like the `language` of a project).

## 3. Orchestrating the Memory Lifecycle
We established a clean dependency injection pipeline that looks like this:
```text
Conversation
      │
MemoryExtractor (Parses text into Candidate Memories)
      │
MemoryManager (The boss. Orchestrates the workflow)
      │
DecisionEngine (Evaluates conflict resolution via Policies)
      │
MemoryRepository (The grunt. Handles CRUD operations to memory.json)
```
The **MemoryManager** handles the core orchestration: it fetches the existing active memory, consults the DecisionEngine, and executes the resulting decision (`STORE`, `IGNORE`, or `UPDATE`).

## 4. Policy-Driven Decision Engine
**The Problem:** Not all memories evolve the same way. A change in a project's language requires an `UPDATE` even if the project name remains identical, while an environment preference only updates if the primary value changes.

**The Solution:** We implemented the **Strategy Pattern**. 
- We built a `PolicyFactory` that routes candidate memories to specific policies (`EnvironmentPolicy`, `ProjectPolicy`, etc.).
- The `DecisionEngine` was reduced to a thin router: `policy = PolicyFactory.get(memory["type"])`.
- All the messy, complicated logic of *how* a memory evolves is shoved into its respective policy class. Out of sight, out of mind.

## 5. Dynamic Persistence
The **MemoryRepository** was upgraded to handle targeted updates. When a policy screams `UPDATE`, the repository finds the matching active memory and selectively merges the new fields (like updating the `language` and bumping the timestamp) while leaving the original `id` intact. 

No more digital hoarding. Just clean state updates.