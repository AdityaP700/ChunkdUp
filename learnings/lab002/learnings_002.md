# Lab 002: The Trust Problem (Structured Outputs)

## The Story So Far...
In Lab 001, we figured out how to search through our data and pack the most relevant chunks into a strict context window. We successfully built the retrieval engine. 

But then we hit the next wall: **Why did we need an LLM in the first place?** 
We needed the LLM to read those chunks and synthesize an actual answer. 

The problem? LLMs are conversational by design. If you feed an LLM your chunks and ask a question, it replies with friendly, rambling paragraphs: *"Sure! Based on the context provided, the answer is..."* 

That's great for a human reading a chatbot screen, but **it is an absolute nightmare for software.** If you inject raw, conversational strings directly into your application, your software will break. We needed the LLM to stop acting like a chatty assistant and start acting like a strict, reliable function that returns predictable data (like the answer, and exactly which chunk it used).

## The Engineering Challenge: Forcing Predictability
Can I trust the LLM's output? No. Of course not. 

So, we had to build a straitjacket for it. To see this in action, open up `data/chunks.json` to see the raw text we feed the system, and run `experiment_001.py`. Here is how we tamed the model:

### 1. The Two-Prompt Approach (Basic vs. Expert)
We built a `PromptBuilder` that doesn't just ask for an answer—it demands a specific JSON schema. We constructed two distinct prompts:
- **Basic:** A standard instruction demanding the answer and a list of citations (e.g., `[1, 6]`).
- **Expert (The Judge):** A much stricter prompt that commands the LLM to use *only* the provided context. If the answer isn't there, it forces the LLM to admit "I don't know." We turned the LLM into a strict judge of its own context rather than a creative writer.

### 2. Resilient Parsing (Because LLMs Rebel)
Even when threatened, LLMs will inevitably wrap their requested JSON in useless markdown fences (```` ```json ````). To survive this, we built an `OutputParser` that uses aggressive regex to hunt down and extract anything resembling curly braces, salvaging the JSON payload from the conversational wreckage.

### 3. Strict Output Validation (The Bouncer)
Extracting the JSON was only half the battle. If the structure is wrong, our app crashes anyway. We built an `OutputValidator` to act as a schema checker:
- **Presence Checks:** If you don't have an `answer` or `confidence`, the pipeline rejects it.
- **Type & Bounds Checking:** If `citations` isn't a list, or if `confidence` is somehow `1.5`, the LLM is clearly hallucinating. Rejected.

## The Ultimate Result
What did we actually achieve? 

We successfully built a **deterministic wrapper around a probabilistic engine.** 

By making validation configurable (`ENABLE_VALIDATION = True`), we proved a critical production engineering principle: **Never trust the model just because it produced valid JSON.** Our system now intentionally rejects non-compliant outputs rather than risking downstream crashes. 

We transformed a hallucination-prone text generator into a reliable software component that returns strict, typed, and verifiable data. Trust no one. Especially the AI.