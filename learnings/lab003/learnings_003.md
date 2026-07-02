# Lab 003: The State Problem (Resilient Memory Pipeline)

## The Story So Far...
By the end of Lab 002, we had a bulletproof pipeline. We could find exactly the right data (Lab 001) and force the LLM to return safe, machine-readable answers (Lab 002). 

But the system suffered from severe amnesia. Every single interaction was treated as a completely isolated event. It didn't know who the user was, what operating system they used, or what project they were building. 

We needed state. We needed memory.

## The Engineering Challenge: The Conflict of Evolving Facts
Extracting facts from text is actually the easy part. The real engineering nightmare starts the second you extract a fact that contradicts something you already know. 

If the user said they use "Windows" yesterday, but today they say they use "Linux", what do you do? Do you keep both? Do you overwrite it? Do you ignore it? If you just blindly append every extracted fact to a JSON file, you aren't building a memory system—you are building a digital hoarding disorder that will eventually crash under its own weight.

Here is how we solved the state problem:

### 1. The Declarative Extractor
First, we threw away our rigid string-parsing hacks and built a declarative `MemoryExtractor`. Using regex rules mapped to specific memory properties, we built a somewhat intelligent ingestion layer. One day we'll swap the regex for an LLM extractor, but for now, it gets the job done without breaking downstream components.

### 2. Rich Metadata Schema
Instead of raw key-value pairs (`{"os": "Windows"}`), memories became robust entities. We enriched them with UUIDs, timestamps, source tracking, and type categorization. A memory wasn't just a string anymore; it was a trackable object with a lifecycle.

### 3. Orchestrating the Chaos
We built a clean pipeline to handle the flow of time:
```text
Conversation -> MemoryExtractor -> MemoryManager -> DecisionEngine -> MemoryRepository
```
The **MemoryManager** became the boss. Whenever a new memory is extracted, it checks the repository for an existing version of that memory, and asks the DecisionEngine what to do (`STORE`, `IGNORE`, or `UPDATE`).

### 4. Policy-Driven State Management (The Strategy Pattern)
We quickly hit a wall: not all memories evolve the same way. A change in a project's `language` requires an `UPDATE` even if the project name remains identical, while an environment preference only updates if the primary value changes.

To solve this, we implemented the **Strategy Pattern**. We built a `PolicyFactory` that routes candidate memories to specific policies (`EnvironmentPolicy`, `ProjectPolicy`, etc.). The `DecisionEngine` was reduced to a thin, clean router. All the messy, complicated logic of *how* a specific type of memory evolves was shoved into its respective policy class. Out of sight, out of mind.

### 5. Dynamic Persistence
Finally, we upgraded the `MemoryRepository` to handle targeted updates. When a policy screams `UPDATE`, the repository doesn't just append a new row. It finds the matching active memory, selectively merges the new fields (like updating the `language` and bumping the timestamp), and leaves the original UUID intact. 

## The Ultimate Result
What did we actually achieve?

We transformed a stateless system into a **resilient, self-managing entity**. We built an engine that doesn't just store data, but actively reasons about state conflicts and resolves them using strict, type-specific policies. 

No more digital hoarding. Just clean, evolving state. 