Can we extract memories without depending on fixed string patterns?

"Eventually we'll replace all of this with either:

-Regular expressions
-spaCy (NER/pattern matching)
-An LLM extracting structured memories
-Function calling / structured extraction"

#Updates:
as of now :moved from a rule-based parser to a declarative extraction engine.

alright now the case is that ,we will be building a MemoryManager

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