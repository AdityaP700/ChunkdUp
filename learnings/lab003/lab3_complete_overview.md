# Lab 003: The Complete Overview (The Memory Pipeline)

If you need to explain Lab 3 to another engineer or in an interview, this document covers everything from the core problem down to the exact mechanical execution of the code we built.

---

## 1. The Core Problem: State and Amnesia
By the end of Lab 2, our AI system was a bulletproof, deterministic machine that could query data and return strict JSON. But it had total amnesia. Every conversation was treated like the first one. 

We needed memory. 

But managing memory isn't just about extracting facts. Extracting facts is easy. The real engineering nightmare is **Conflict Resolution**. If a user says they use "Windows" on Monday, but on Tuesday they say they use "Linux", what do you do? If you blindly append every extracted fact to a JSON database, you create a bloated, contradictory mess. You need a system that can gracefully evolve state over time.

---

## 2. The Architecture Pipeline
To solve this, we built a strict, decoupled pipeline. No single component knows how the whole system works; each just does its specific job.

```text
Conversation 
      │
MemoryExtractor (Parses text, attaches metadata)
      │
MemoryManager (The Orchestrator)
      │
DecisionEngine (Routes to PolicyFactory)
      │
MemoryRepository (Executes CRUD operations on memory.json)
```

---

## 3. Step One: The Memory Extractor
When a user types *"I am building ChunkdUp in Python"*, the `MemoryExtractor` intercepts it. 

Right now, we use a **Regex Parser**. It searches for patterns (e.g., `I'm building ([\w\d\-]+) in ([\w\d\+#]+)`) and forces the raw text into a strict schema. Most importantly, it attaches a `"type"` (e.g., `project`, `environment`, `tool`).

**The Limitation:** Regex is brittle because it lacks semantic intent. If a user says *"I hate using Windows"*, the regex might accidentally match `"use Windows"` and log it as a fact. This is perfectly fine for proving our architecture works, but in production, you swap this Regex layer for an LLM Extractor that actually understands what the user meant.

---

## 4. Step Two: The Policy Engine (Strategy Pattern)
Once the memory is extracted, the `MemoryManager` asks the database (`MemoryRepository`): *"Do we already know anything about this specific key?"* It then hands both the old memory and the new memory to the `DecisionEngine`.

We did **not** use a giant `if/else` block here, and we did **not** ask an LLM to resolve the conflict. LLMs are too slow and expensive for state management, and `if/else` blocks turn into unmaintainable spaghetti code.

Instead, we used the **Strategy Pattern**. Think of it like a hospital triage desk:
1. **The Manager (Triage Nurse):** Receives the new patient (memory) and grabs their old file. 
2. **The Engine (Dispatcher):** Looks at the memory `"type"` and routes it to a Specialist.
3. **The Policies (Specialist Doctors):** We built isolated classes like `ProjectPolicy` and `EnvironmentPolicy`. The policy looks at the old and new data and makes a strict decision: `STORE`, `IGNORE`, or `UPDATE`.

By separating memories into buckets (`Environment > Tool > Preference > Project`), the system understands that changing your font color (Preference) requires totally different logic than migrating your codebase from Python to Rust (Project).

---

## 5. Step Three: The Mechanics of Resolution
Based on what the Policy decides, here is exactly what the system does mechanically:

- **Action: `STORE`** 
  - *Trigger:* No prior memory was found.
  - *Execution:* The Manager attaches a UUID and a timestamp, and tells the Repository to write a brand new row in `memory.json`.
- **Action: `IGNORE`** 
  - *Trigger:* The old memory and the new memory are completely identical.
  - *Execution:* The Manager literally drops the new memory in the trash. The database is never touched, preventing disk-write bloat.
- **Action: `UPDATE`** 
  - *Trigger:* A matching record exists, but the facts changed (e.g., Windows -> Linux).
  - *Execution:* The Manager does **not** create a new row. It tells the Repository to find the original record by its UUID, merge the new value into it, and bump the `updated_at` timestamp. 

---

## 6. Proving it Works (The Test Cases)
If you look at the 4 test cases we built in `test_policies.py`, they prove this architecture flawlessly handles the four core scenarios of state mutation:

1. **Brand New Data:** `(None, New Windows) -> STORE`
   *No record existed, so we created one.*
2. **Redundant Data:** `(Old Windows, New Windows) -> IGNORE`
   *Nothing changed, so we ignored it to save space.*
3. **Mutated Data:** `(Old Windows, New Linux) -> UPDATE`
   *The OS changed, so we overwrote the old record to maintain a single source of truth.*
4. **Orthogonal Data:** `(Old Windows, New VS Code) -> STORE`
   *An editor does not replace an OS. They are different categories, so we stored it alongside the OS.*

---

## 7. Adding the Memory Scorer (Filtering the Noise)
Before a memory reaches the Decision Engine, we realized not all memories are created equal. A project memory ("I'm building ChunkdUp") is highly important, but a random question might just be noise.

We introduced the `MemoryScorer`—a deterministic component inserted between the Extractor and the Decision Engine. It assigns an `importance` score (e.g., project -> 1.0, environment -> 0.8) and applies a **threshold** (e.g., 0.6). Memories below the threshold are dropped immediately, saving the repository from processing garbage. 

---

## 8. Repetition as Evidence (`MERGE` Action)
Initially, if a user repeated a fact ("I use Linux"), our policies triggered an `IGNORE` decision. The problem? We were throwing away valuable signal. 

We evolved `IGNORE` into `MERGE`. When a duplicate fact is detected, the Manager now triggers the `MemoryRepository.merge()` operation. Instead of ignoring the fact, it finds the existing active memory and increments its `"frequency"` counter. 

What used to be considered "redundant noise" is now captured as **evidence**. A memory with `frequency: 3` gives the AI high confidence that this is a deeply ingrained user preference, rather than an offhand comment.

## 9. Contradiction & State Mutation (`UPDATE` Action)
If a user contradicts a previous fact (e.g., "I use Windows", followed later by "I use Linux"), the pipeline triggers an `UPDATE`. To test this, we built a dedicated testing hierarchy (`tests/test-contradiction.py`) that revealed four core engineering truths about our mechanical update loop:
1. **Frequency resets:** When the state mutates, the `frequency` is mechanically forced back to `1`. A new fact should not inherit the high confidence (frequency) of the old fact.
2. **`updated_at` changes:** The update timestamp is accurately bumped to reflect the exact moment of the contradiction.
3. **`created_at` remains:** The original timestamp when the memory key (e.g., `os`) was discovered is strictly preserved.
4. **History is overwritten:** Currently, the system performs an in-place overwrite. The old state ("Windows") is lost. For a production system requiring audit trails, this is a dangerous design that points toward future improvements (e.g., archiving old records rather than deleting them).

---

## 10. What's Next?
We have a robust state-management pipeline with noise filtering, frequency tracking, and clean state mutation. The next frontier is moving away from brittle Regex Extractors and hardcoded Heuristic Scorers toward robust, dynamic LLM-driven components that can understand true semantic intent without sacrificing our deterministic architecture.
