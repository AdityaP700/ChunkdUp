## failure with previous approach
Hardcoded JSON	Can't use PostgreSQL, SQLite, or cloud storage
No IGNORE	Stored everything, even useless info
No UPDATE	Couldn't change "Python" to "Go"
No evaluation	Didn't know if it actually worked
No frequency	Didn't track how often something was mentioned
No recency	Old and new memories were treated equally
No scoring	All memories had same importance
No type safety	Prone to bugs, no IDE help
No docs	Hard for others to use

We Wanted to Achieve
✅ Make it usable by others (SDK)
✅ Make it intelligent (decisions)
✅ Make it measurable (evaluation)
✅ Make it production-ready (PostgreSQL)
✅ Make it reliable (observability)


#failure for the imposition of the type filtering and the threshold

## Failure 1: The "Weather" Problem
Query	"What is the weather?"
Expected	No results (or "I don't know")
Actual	Returned 5 memories about programming, editors, dark mode
Root Cause	Embedding model finds semantic connection between "weather" and "dark mode", "IDE", "environment"
Impact	User gets completely irrelevant results

## Failure 2: The "Language" Priority Problem
Query	"What programming languages do I use?"
Expected	Python, Go (in that order)
Actual	ChunkdUp, Neovim, Google, dark mode (no Python/Go in top 5)
Root Cause	Semantic similarity treats all "programming context" equally
Impact	Most relevant memories buried under less relevant ones

## Failure 3: The "Tech Stack" Dilution Problem
Query	"What is my tech stack?"
Expected	Python, Go, Neovim (languages + tools)
Actual	Neovim, Google, ChunkdUp (project prioritized)
Root Cause	No type-based weighting
Impact	Contextually relevant but not prioritized by type

## Failure 4: The "Exact Match" Problem (Already Solvable)
Query	"Python"
Expected	Memory with "Python"
Actual	Found via keyword search ✅
Root Cause	N/A (keyword search already handles this)
Impact	Not a failure, but hybrid search should maintain this

### Options to be explored :
## Option A: Similarity Threshold
- What It Does: Only return results above a certain similarity score.
 but it doesnt  type
### Type-Based Filtering/Boost
What It Does: Boosts memories based on their type.
we were boasting the priority of the type of the memories ,but it could be vague
- Weather	❌ Doesn't fix	Still returns irrelevant results
### Option C: Query Type Detection
What It Does: Detect the type of query and filter results accordingly.

- but it doesnt guarantee exact matches

### Option D: Hybrid (Keyword + Semantic with Reranking)

Test	Query	Expected	Found?	Rank	Verdict
Exact: Where runs?	"Where does ChunkdUp run?"	runtime_environment	❌	N/A	FAILED
Exact: Database?	"What database?"	database	✅	1	PASSED
Exact: Embedding model?	"What embedding model?"	embedding_model	❌	N/A	FAILED
Exact: Language?	"What language?"	programming_language	✅	1	PASSED
Semantic: Hosted?	"Where is ChunkdUp hosted?"	runtime_environment	❌	N/A	FAILED
Semantic: Vector search?	"What technology for vector search?"	database	❌	N/A	FAILED
Semantic: Model?	"What model for embeddings?"	embedding_model	❌	N/A	FAILED
Semantic: Code?	"What language for backend?"	programming_language	✅	1	PASSED

aditya07@NitroAdi:/mnt/d/Development/Projects/core/ChunkdUp$ docker compose exec app python evaluation/test_semantic_retrieval.py
INFO:chunkdup.memory:Memory initialized with store: postgres
Seeding memories for testing...
INFO:chunkdup.memory:Memory cleared
INFO:chunkdup.memory:Remembering: I use Python for backend development....
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: /opt/models/all-MiniLM-L6-v2
INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: cpu
INFO:chunkdup.embeddings:Loaded embedding model: /opt/models/all-MiniLM-L6-v2
Batches: 100%|██████████████████████████████████████| 1/1 [00:00<00:00,  3.02it/s]
INFO:chunkdup.memory:Stored new memory: programming_language = Python
INFO:chunkdup.memory:Remembering: ChunkdUp uses PostgreSQL with pgvector....
Batches: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 35.10it/s]
INFO:chunkdup.memory:Stored new memory: database = PostgreSQL
INFO:chunkdup.memory:Remembering: ChunkdUp runs inside Docker on WSL2....
Batches: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 25.99it/s]
INFO:chunkdup.memory:Stored new memory: runtime_environment = Docker on WSL2
INFO:chunkdup.memory:Remembering: The embedding model is all-MiniLM-L6-v2....
Batches: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 87.73it/s]
INFO:chunkdup.memory:Stored new memory: embedding_model = all-MiniLM-L6-v2
✅ Seeded 4 memories

INFO:chunkdup.memory:Memory initialized with store: postgres
INFO:chunkdup.memory:Searching: Where does ChunkdUp run?
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: /opt/models/all-MiniLM-L6-v2
INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: cpu
INFO:chunkdup.embeddings:Loaded embedding model: /opt/models/all-MiniLM-L6-v2
Batches: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 30.64it/s]
INFO:chunkdup.memory:Searching: What database does ChunkdUp use?
Batches: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 18.30it/s]
INFO:chunkdup.memory:Searching: What embedding model?
Batches: 100%|█████████████████████████████████████| 1/1 [00:00<00:00, 134.29it/s]
INFO:chunkdup.memory:Searching: What language?
Batches: 100%|█████████████████████████████████████| 1/1 [00:00<00:00, 117.27it/s]
INFO:chunkdup.memory:Searching: Where is ChunkdUp hosted?
Batches: 100%|█████████████████████████████████████| 1/1 [00:00<00:00, 108.78it/s]
INFO:chunkdup.memory:Searching: What technology for vector search?
Batches: 100%|█████████████████████████████████████| 1/1 [00:00<00:00, 110.05it/s]
INFO:chunkdup.memory:Searching: What model for embeddings?
Batches: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 89.25it/s]
INFO:chunkdup.memory:Searching: What language for backend?
Batches: 100%|█████████████████████████████████████| 1/1 [00:00<00:00, 107.78it/s]

======================================================================
        CHUNKDUP — SEMANTIC RETRIEVAL TEST SUITE
======================================================================

❌ Exact: Where runs?
   Query: "Where does ChunkdUp run?"
   Expected: runtime_environment = Docker
   Found: False (Rank: N/A)
   Top 3: [('programming_language', 'Python', 0.6674676337876756), ('database', 'PostgreSQL', 0.5706852930375769), ('runtime_environment', 'Docker on WSL2', 0.47721004757761387)]

✅ Exact: Database?
   Query: "What database does ChunkdUp use?"
   Expected: database = PostgreSQL
   Found: True (Rank: 1)
   Top 3: [('database', 'PostgreSQL', 0.7643700838088989), ('programming_language', 'Python', 0.6400517439811394), ('embedding_model', 'all-MiniLM-L6-v2', 0.44800758361816406)]

✅ Exact: Embedding model?
   Query: "What embedding model?"
   Expected: embedding_model = all-MiniLM-L6-v2
   Found: True (Rank: 1)
   Top 3: [('embedding_model', 'all-MiniLM-L6-v2', 0.5550102591514587), ('programming_language', 'Python', 0.07415784453404894), ('database', 'PostgreSQL', 0.06782636046409607)]

✅ Exact: Language?
   Query: "What language?"
   Expected: programming_language = Python
   Found: True (Rank: 1)
   Top 3: [('programming_language', 'Python', 0.2750528968854886), ('embedding_model', 'all-MiniLM-L6-v2', 0.1418627989847303), ('database', 'PostgreSQL', 0.022904670548578254)]

❌ Semantic: Hosted?
   Query: "Where is ChunkdUp hosted?"
   Expected: runtime_environment = Docker
   Found: False (Rank: N/A)
   Top 3: [('programming_language', 'Python', 0.5643877310290657), ('database', 'PostgreSQL', 0.5544115570538318), ('runtime_environment', 'Docker on WSL2', 0.410072695008195)]

❌ Semantic: Vector search?
   Query: "What technology for vector search?"
   Expected: database = PostgreSQL with pgvector
   Found: False (Rank: N/A)
   Top 3: [('programming_language', 'Python', 0.09971777500404289), ('embedding_model', 'all-MiniLM-L6-v2', 0.09297755360603333), ('database', 'PostgreSQL', 0.025516483932733536)]

✅ Semantic: Model?
   Query: "What model for embeddings?"
   Expected: embedding_model = all-MiniLM-L6-v2
   Found: True (Rank: 1)
   Top 3: [('embedding_model', 'all-MiniLM-L6-v2', 0.5350760221481323), ('programming_language', 'Python', 0.10475878538138972), ('database', 'PostgreSQL', 0.07734252512454987)]

✅ Semantic: Code?
   Query: "What language for backend?"
   Expected: programming_language = Python
   Found: True (Rank: 1)
   Top 3: [('programming_language', 'Python', 0.27727274434132965), ('database', 'PostgreSQL', 0.10713493824005127), ('runtime_environment', 'Docker on WSL2', 0.07219601737284576)]

======================================================================
Passed: 5
Failed: 3
 Accuracy: 62.5%