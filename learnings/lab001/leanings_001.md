# Lab 001: The Constraint Problem

## The Big Annoying Question
How should an AI system actually decide what information reaches the model, considering context windows aren't infinite and tokens actually cost money?

## What We Messed Around With (Experiments)
- **Keyword Retrieval:** Counting matching words. (Terrible idea. Punctuation breaks it).
- **Semantic Retrieval:** Actually understanding meaning via embeddings. (Much better).
- **Top-K Assembly:** Just grabbing the top 3 results blindly. (Also a terrible idea, because chunks have different lengths and you will blow up your context window).
- **Budget-Aware Assembly:** Packing chunks based on an actual character limit. (A surprisingly sane approach).
- **Prompt Variants:** Trying to get the LLM to behave predictably.

## The Harsh Realities (Decisions)
- Semantic retrieval completely obliterated keyword matching. We're never looking back.
- Budget-aware assembly is far superior to fixed Top-K. Blindly grabbing 3 chunks is a great way to either crash your pipeline or waste context space.
- If an oversized chunk doesn't fit the remaining budget, just skip it and grab the next one. Don't halt the whole assembly process just because one chunk is too fat.
- Prompt formatting is totally separate from retrieval. Keep your concerns separated, folks.
- Structured JSON outputs are non-negotiable if you want LLM responses to be actually usable by a machine.

## Stuff That Still Bothers Me (Open Questions)
- When should reranking actually happen?
- Should we just brutally filter out low-score chunks even if we have budget left? (Garbage in, garbage out, right?)
- Where does compression fit into this mess?