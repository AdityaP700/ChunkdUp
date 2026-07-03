# Interview Prep: How We Group Retrieved Chunks into Prompts

**Common Interview Question:** *"Once you retrieve your context chunks, how do you actually format or group them in the prompt so the LLM understands them?"*

This bridges the gap between **Lab 1 (Context Assembly)** and **Lab 2 (Structured Outputs)**. Here is exactly how we solve it, and how you should explain it in an interview.

---

## 1. The Core Problem: LLMs Don't Read Arrays
When our `ContextAssembler` finishes its job in Lab 1, it hands us a list of Python dictionaries (e.g., `[{"text": "foo", "score": 0.9}, {"text": "bar", "score": 0.8}]`). 

LLMs do not understand Python dictionaries. An LLM reads a single, flat, one-dimensional string of tokens. If you just dump a raw JSON array of chunks into a prompt, you waste valuable tokens on curly braces and syntax, and you confuse the model. We have to **linearize** the data.

## 2. The Solution: The `PromptBuilder`

In Lab 2, we built a `PromptBuilder` class. Its job is to take that array of chunks and compile it into a highly structured string. Here is exactly how we group the data:

### A. Enumeration (Numbering the Chunks)
We don't just concatenate the text. We loop through the chunks and explicitly number them:
```text
1. Chunk text goes here...
2. Another chunk text goes here...
3. Final chunk text goes here...
```
**Why?** This is critical for Lab 2 (Structured Outputs). Because we force the LLM to return a JSON array of `"citations"`, the LLM needs a way to reference the data. By numbering the chunks, the LLM can easily say, *"I got this answer from chunks [1, 3]."* If you don't number them, citations become impossible.

### B. Strict Bounding (Delimiters)
We inject that numbered list into the prompt, but we wrap it in strict visual delimiters:
```text
Context:
----------------
1. Chunk text...
2. Chunk text...
----------------
```
**Why?** LLMs are highly susceptible to "instruction confusion" (or prompt injection). If a chunk of text happens to contain a sentence like *"Ignore all previous instructions,"* the LLM might execute it. By placing the chunks inside strict delimiters (`----------------`), and explicitly telling the model *"Answer the question using ONLY the provided context below"*, we create a clear boundary between **Instructions** and **Data**.

### C. The Graceful Fallback
If the retriever found absolutely nothing (an empty list), we don't just leave the context block blank. The `PromptBuilder` intercepts it and injects a hardcoded string: *"No relevant context found."*
**Why?** If you leave it blank, the LLM will usually hallucinate an answer using its pre-trained weights. By explicitly injecting "No context found," combined with our Expert Prompt instruction (*"If the answer is not present, say I don't know"*), we guarantee the system fails gracefully instead of lying to the user.

---

## How to Pitch This in an Interview

**"To format retrieved context, I built a `PromptBuilder` that acts as an isolation layer between the raw data and the LLM. It linearizes the retrieved chunks into an enumerated string. The enumeration is critical because it allows the LLM to return deterministic citations (e.g., chunk 1 and 3). Furthermore, I wrap the concatenated chunks in strict visual delimiters inside the prompt template. This clearly separates system instructions from raw data, which significantly reduces the risk of the LLM getting confused by the payload or hallucinating."**
