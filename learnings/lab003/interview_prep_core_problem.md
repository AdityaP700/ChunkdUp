# Interview Prep: Managing AI Memory Constraints

When discussing the architecture of an AI system (like the one built in Lab 3) in an interview, here is how you can clearly articulate the core problem, the goal, and why the naive solution fails.

## 1. The Core Problem: Memory Constraints & State

**If they ask:** *"What was the core problem you were trying to solve?"*

**How to say it in an interview:**
> *"The core engineering challenge in building conversational AI agents is managing state under strict memory constraints. LLMs are stateless by default and possess finite, expensive context windows. We can't simply feed the entire conversation history back into the model on every request—it's computationally expensive, increases latency, and eventually hits hard token limits. 
>
> Furthermore, the problem isn't just about extracting facts to save space; it's about **conflict resolution**. If a user says they use 'Windows' on Monday, but 'Linux' on Tuesday, a basic system will store both. The real challenge is building a deterministic system that can gracefully resolve state conflicts over time, ensuring the context remains lean, accurate, and free of contradictions before it ever reaches the LLM."*

---

## 2. The Goal

**What we set out to build:**
The goal was to design a **decoupled, deterministic memory pipeline** that intercepts user input, extracts structured facts, and routes them through a Policy Engine. 

By using the **Strategy Pattern**, the system categorizes memories (e.g., Environment, Project, Tool) and applies specific logic to either `STORE` new data, `IGNORE` redundant data, or `UPDATE` mutated data. This ensures we maintain a single source of truth—a clean, highly compressed JSON state—without wasting LLM tokens on basic database CRUD operations.

---

## 3. The Naive Solution (And Why It Fails)

**The Naive Approach:**
The most common beginner approach is to blindly append every extracted fact (or worse, raw chat logs) into a database array, and then inject that massive, unfiltered list into the LLM's system prompt on the next turn. Another naive approach is asking the LLM itself to manage its own memory and resolve conflicts during the conversation.

**Why it fails in production:**
1. **Context Bloat (Cost & Latency):** Appending everything rapidly consumes the context window, driving up API costs and drastically slowing down response times.
2. **Contradictions:** If the database contains both *"User uses Windows"* and *"User uses Linux"*, the LLM gets confused, leading to hallucinations or degraded reasoning.
3. **Non-Deterministic:** Relying on the LLM to figure out which fact is the "current" one is unpredictable. State management (CRUD operations) should be handled by fast, deterministic code (like Python), not expensive probabilistic models.
