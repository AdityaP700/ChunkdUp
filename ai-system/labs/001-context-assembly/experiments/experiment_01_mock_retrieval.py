chunks =[
    {
        "id": 1,
        "score": 0.95,
        "text": "Context Engineering is deciding what information reaches the LLM.",
        "keywords": ["context", "engineering", "llm", "information"]
    },
    {
        "id": 2,
        "score": 0.91,
        "text": "Token budgets are limited.",
        "keywords": ["token", "budget", "limited"]
    },
    {
        "id": 3,
        "score": 0.88,
        "text": "Retrieval returns candidate chunks.",
        "keywords": ["retrieval", "candidate", "chunks"]
    },
    {
        "id": 4,
        "score": 0.87,
        "text": "Reranking improves the quality of selected context.",
        "keywords": ["reranking", "quality", "context"]
    },
    {
        "id": 5,
        "score": 0.82,
        "text": "Long contexts can degrade model performance.",
        "keywords": ["context", "performance", "long"]
    },
    {
        "id": 6,
        "score": 0.79,
        "text": "Irrelevant chunks should be filtered out before reaching the model.",
        "keywords": ["irrelevant", "filter", "model"]
    },
    {
        "id": 7,
        "score": 0.76,
        "text": "Preserving document order can sometimes be more important than score.",
        "keywords": ["order", "document", "score"]
    },
    {
        "id": 8,
        "score": 0.71,
        "text": "Chunking strategy affects retrieval quality significantly.",
        "keywords": ["chunking", "retrieval", "quality"]
    },
    {
        "id": 9,
        "score": 0.68,
        "text": "Metadata filtering helps reduce noise in retrieved results.",
        "keywords": ["metadata", "filtering", "noise"]
    },
    {
        "id": 10,
        "score": 0.65,
        "text": "Some high-scoring chunks may still be irrelevant to the query.",
        "keywords": ["irrelevant", "query", "score"]
    },
    {
        "id": 11,
        "score": 0.61,
        "text": "Context window size is a hard constraint in most LLMs.",
        "keywords": ["context", "window", "llm"]
    },
    {
        "id": 12,
        "score": 0.55,
        "text": "Diversity in retrieved chunks can improve answer quality.",
        "keywords": ["diversity", "quality", "retrieved"]
    }
]
class Retriever:
    def __init__(self, chunks_list):
        self.chunks = chunks_list

    #measures how relevant a chunk is to the query by counting how many words
    #in common between them
    def _calculate_overlap(self, query: str,chunk_keywords:list)->int:
        query_words=set(query.lower().split())
        chunk_words=set(word.lower() for word in chunk_keywords)
        return len(query_words & chunk_words)

    #i call retriever.retrieve(query,k)
    #it creates an empty list called scored_chunks=[]
    def retrieve(self, query: str = "", k: int = 10):
        """Always returns exactly k chunks by cycling through available ones."""
        if not self.chunks:
            return []
        scored_chunks=[]
        #for each chunk,it calls the _calculate...it gets back a number

        for chunk in self.chunks:
            #it creates a tuple (overlap,original_score,chunk)
            overlap=self._calculate_overlap(query,chunk.get("keywords",[]))
            #it adds this tuple to scored_chunk
            scored_chunks.append((overlap,chunk["score"],chunk))

        #it sorts scored_chunks
        scored_chunks.sort(key=lambda x:(x[0],x[1]),reverse=True)
        #it takes the top k chunks from the sorted list
        top_chunks=[chunk.copy() for _, _, chunk in scored_chunks[:k]]
        return top_chunks

class ContextAssembler:
    def __init__(self):
        pass
    
#it doesnt takes any data ,it just takes the chunks
#input when assemble() is called
    def assemble(self,chunks:list[dict])->list[dict]:

        if not chunks:
            return []

        sorted_chunks=sorted(
            chunks,
            #konsa cheez ko sort out karna hai
            #kis tarike se karna hai
            #aur kya reverse hoga??nhi
            key=lambda x : x.get("score",0.0),
            reverse=True
        )

        return sorted_chunks[:3]

question ="what is diversity in retrieval?"

retriever = Retriever(chunks)
assembler = ContextAssembler()

retrieved_chunks = retriever.retrieve(query=question,k=10)
final_context = assembler.assemble(retrieved_chunks)

print(f"Received {len(retrieved_chunks)} chunks from the retrieval")
print(f"Selected {len(final_context)} chunks for the model\n")

for i, chunk in enumerate(final_context,1):
    print(f"{i}.[scores={chunk['score']}] {chunk['text']}")
