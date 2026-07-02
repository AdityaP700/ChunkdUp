# Lab 001: The Constraint Problem (Context Assembly)

## The Story So Far...
We wanted to build an AI that actually knows things about our specific data. The naive approach is just to shove your entire database into the LLM prompt. 

The problem? Context windows aren't infinite, and even if they were, shoving 10,000 pages of text into a prompt is incredibly expensive and mathematically guaranteed to confuse the model. We needed a gatekeeper. We needed a system to decide *exactly* what small slice of information gets the privilege of reaching the model.

## The Engineering Challenge: Filtering the Noise
How do we find the right information, and how do we fit it into a strict space limit? Here is the painful evolution of our pipeline:

### 1. Keyword Retrieval (The Dumb Search)
First, we tried counting matching words. If the user asked about "Python," we looked for chunks with the word "Python." 
**The Reality Check:** It was a terrible idea. Punctuation broke it. Synonyms completely bypassed it. It had zero understanding of actual meaning. We threw it out and moved to **Semantic Retrieval**, using dense vector embeddings to actually match the *intent* of the query rather than just the letters.

### 2. Top-K Assembly (The Blind Grab)
Once we had semantic search working, we had to assemble the prompt. The standard industry tutorial says, "Just grab the Top 3 results!" So we tried it.
**The Reality Check:** Also a terrible idea. Chunks have wildly different lengths. Sometimes the top 3 chunks are 50 characters each. Sometimes they are 5,000 characters each. Blindly grabbing 3 chunks is a fantastic way to either completely waste your context window or blow it up and crash your pipeline.

### 3. Budget-Aware Assembly (The Tetris Approach)
We realized we were treating text like abstract concepts instead of what it really is: physical data with a strict size limit. We built a `ContextAssembler` that acts like a greedy packer. You give it a strict character budget (e.g., 800 chars). It iterates through the semantically ranked chunks and adds them one by one. If it hits an oversized chunk that won't fit, it doesn't halt the whole process—it just skips it and grabs the next smaller one that fits perfectly. 

## The Ultimate Result
What did we actually achieve? 

We built a **bulletproof context pipeline**. We successfully separated the logic of *finding* data from the logic of *packing* data. Our system now searches by actual meaning, and gracefully packs the results into a strictly constrained budget without ever crashing the LLM context window. 

We conquered the constraint problem. But we soon realized that just because we fed the LLM the right data, didn't mean it would give us a usable answer... (which led us straight into the nightmare of Lab 002).