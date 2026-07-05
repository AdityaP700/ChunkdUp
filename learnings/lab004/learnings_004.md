# Lab 4: The Semantic Memory Pipeline

The core question of Lab 4 is: **Out of 10,000 stored memories, which ones deserve to enter the context window, and how are they presented?**

We are transitioning from a simple memory database to a fully functional, memory-aware AI assistant by wiring together the architectures of Lab 1, Lab 2, and Lab 3 into one cohesive system.

## The Additive Pipeline Architecture

A key engineering philosophy established here is that **features accumulate**. We do not discard previous labs; we stack them. 

The unified pipeline now looks like this:
```text
Extractor
   ↓
Repository (Lab 3)
   ↓
Retriever (Lab 4)
   ↓
Prompt Builder (Lab 1)
   ↓
LLMCaller (Lab 2)
   ↓
OutputParser (Lab 2)
   ↓
OutputValidator (Lab 2)
```

## Retrieval vs Assembly

Just like Google Search, retrieving data is only half the battle. 
1. **The MemoryRetriever** pulls the most relevant facts as Python objects: `[{"key": "editor", "value": "Neovim"}]`.
2. **The PromptBuilder** acts as the UI. It converts those Python dictionaries into a strictly formatted text prompt so the LLM can easily consume them alongside massive document chunks.

## Verifying the System (Test Scenarios)

We proved that the entire lifecycle (Store -> Update -> Merge -> Retrieve -> Generate) works flawlessly by running three critical tests:

### Test 1: Preventing Hallucinations (Unknown Memory)
**Scenario:** Ask the LLM about a database preference when only the editor is stored.
**Result:** Because the Retriever returned nothing relevant, the LLM defaulted to its strict fallback:
```json
{
  "answer": "I don't know.",
  "confidence": 1.0,
  "citations": []
}
```
**Learning:** Our prompt constraints hold perfectly, completely preventing hallucinations on unknown user state.

### Test 2: Synthesizing Context (Multiple Memories)
**Scenario:** Ask the LLM "What development setup do I use and what project am I building?" while providing Linux, Neovim, and ChunkdUp memories.
**Result:** 
```json
{
  "answer": "You use Neovim as your editor on Linux, and you are building a project called ChunkdUp.",
  "confidence": 1.0,
  "citations": [1, 2, 3]
}
```
**Learning:** The LLM can successfully synthesize multiple disparate memories into a natural conversational response, citing every single fact accurately.

### Test 3: Resolving State Conflicts (Contradiction)
**Scenario:** Tell the system "I use Neovim", then "Actually, I prefer VS Code". Then ask "Which editor do I use?"
**Result:**
```json
{
  "answer": "VS Code",
  "confidence": 1.0,
  "citations": [1]
}
```
**Learning:** The Lab 3 Decision Engine successfully `UPDATE`d the conflicting state. The Memory Retriever ignored the outdated "Neovim" fact, ensuring that only the ground-truth "VS Code" memory entered the context window.

---
By completing these tests, we have successfully crossed the boundary from a static database to an autonomous, memory-aware assistant.
