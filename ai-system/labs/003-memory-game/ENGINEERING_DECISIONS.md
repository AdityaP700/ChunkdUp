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