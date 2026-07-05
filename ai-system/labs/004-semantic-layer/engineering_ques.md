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



well now once we have built the retrieval logic ,we need to decide how that information is presented to the LLMs

Think of Google Search
When you search
"Python decorators"
Google's retriever finds billions of candidate pages.
Does Google just dump all the HTML into your browser?
No.
It builds a page like

Result 1
Title
Snippet

Result 2
Title
Snippet

The search engine retrieved.

The UI assembled.
Exactly the same thing is happening here.

here MemoryRetriever answers

Which memories are relevant?

Example:

Query:
Which editor do I use?

↓

Retrieved

editor = Neovim

OS = Linux

Project = ChunkdUp

Notice something.

This is Python objects.

[
    {"key":"editor","value":"Neovim"},
    {"key":"os","value":"Linux"},
    {"key":"project_name","value":"ChunkdUp"}
]

The LLM cannot consume Python dictionaries.

Somebody has to convert them into a prompt.

That's PromptBuilder's job

PromptBuilder asks

Among the retrieved information...

What should actually be injected?

That's context assembly.

well now :we crossed the boundary from a memory database to a memory-aware assistant.

Test - 1 :
=== ACTUAL VALIDATED JSON ===
{
  "answer": "I don't know.",
  "confidence": 1.0,
  "citations": []
}

[PASSED]: LLM successfully prevented hallucination.
[PASSED]: Citations are empty as expected.

expected.
Our prompt constraints are holding perfectly! Because the Memory Retriever didn't return anything relevant about databases, the LLM defaulted strictly to the I don't know fallback with empty citations.


Current:

Extractor
↓

Repository
↓

Retriever
↓

Prompt Builder
↓

LLM
↓

Parser
↓

Validator