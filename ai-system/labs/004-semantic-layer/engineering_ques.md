The question is that ,

Out of 10,000 stored memories, which ones deserve to enter the context window?

We are transitioning from a deterministic scoring system to a learned, semantic scoring system.

The pipeline becomes

Conversation
        │
        ▼
Memory Repository
        │
        ▼
Semantic Retrieval
        │
        ▼
Top K Memories
        │
        ▼
Prompt Builder
        │
        ▼
LLM


 the first experiment is about
 implementing the persistence layer :
  as in the question b

  Query

↓

Retriever

↓

Relevant memories

--- Test 1 ---
Query: Which editor do I use?
Expected: 1. editor Neovim score=1

--- Test 2 ---
Query: What project am I building?
Expected: ChunkdUp (score=1)
