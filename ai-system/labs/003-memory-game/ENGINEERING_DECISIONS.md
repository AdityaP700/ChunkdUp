Can we extract memories without depending on fixed string patterns?

"Eventually we'll replace all of this with either:

-Regular expressions
-spaCy (NER/pattern matching)
-An LLM extracting structured memories
-Function calling / structured extraction"

#Updates:
as of now :moved from a rule-based parser to a declarative extraction engine.

alright now the case is that ,we will be building a MemoryManager

Conversation
     │
MemoryExtractor
     │
Candidate Memories
     │
MemoryManager
     │
MemoryRepository
     │
memory.json

where the job would be simple :
the candidate memory -> should i
- store?/
- update???
- ignore??
- delete??

\
alright currently i am creating a new metaData fields where there would be addition of
- created_at
- updated_at
- confidence
- status
- source
- id

Question
"Can the system persist extracted memories?"
- it could persist the extracted memories


alright the next question comes
how ,when and what polices to modify ,discard or add??

New Memory
↓
Memory Type
↓
Policy
↓
Decision
↓
Repository

Dreaming is a periodic maintenance job!!

Conversation
      │
Memory Extractor
      │
Memory Event
      │
────────────────────────
MemoryManager
────────────────────────
      │
Duplicate Detector
      │
Conflict Detector
      │
Policy Engine
      │
Decision
      │
Repository


now the case is
Existing Memory

+

New Memory

↓

Decision Engine

↓

Decision


The DecisionEngine exists...

...but it isn't actually being used yet.

hence the new decision architecture

New Memory
      │
Repository.get_all()
      │
Find matching memory
      │
DecisionEngine.decide()
      │
Execute decision


what i built till now
Conversation
      │
      ▼
MemoryExtractor
      │
      ▼
Candidate Memories
      │
      ▼
DecisionEngine
      │
      ▼
Decision
      │
      ▼
MemoryRepository
      │
      ▼
memory.json

#Limitations
our first real engineering limitation.

Your current model is

key
value

But your memory is richer.

{
    "key": "project_name",
    "value": "ChunkdUp",
    "language": "Python"
}

so now ,instead of treating every memory the same, we'll introduce type-specific policies.



well experiment till "Can the system decide whether to store, ignore, or update a memory?"
 completed



## New experiment
whether every type of memory follow the same update policy?
my hypothesis : No. Different memory types require different update policies.

lets gooooo!!

so now the game will be based upon the different policies dealing

- "Given this type of memory, how should it evolve?"
i want to change it into

Conversation
        │
MemoryExtractor
        │
MemoryManager
        │
Policy Engine
        │
specific policy
        | 
DecisionEngine
        │
Repository


