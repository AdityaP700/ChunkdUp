chunks =chunks = [
    {
        "id": 1,
        "score": 0.95,
        "text": "Context Engineering is deciding what information reaches the LLM."
    },
    {
        "id": 2,
        "score": 0.91,
        "text": "Token budgets are limited."
    },
    {
        "id": 3,
        "score": 0.88,
        "text": "Retrieval returns candidate chunks."
    },
    {
        "id": 4,
        "score": 0.87,
        "text": "Reranking improves the quality of selected context."
    },
    {
        "id": 5,
        "score": 0.82,
        "text": "Long contexts can degrade model performance."
    },
    {
        "id": 6,
        "score": 0.79,
        "text": "Irrelevant chunks should be filtered out before reaching the model."
    },
    {
        "id": 7,
        "score": 0.76,
        "text": "Preserving document order can sometimes be more important than score."
    },
    {
        "id": 8,
        "score": 0.71,
        "text": "Chunking strategy affects retrieval quality significantly."
    },
    {
        "id": 9,
        "score": 0.68,
        "text": "Metadata filtering helps reduce noise in retrieved results."
    },
    {
        "id": 10,
        "score": 0.65,
        "text": "Some high-scoring chunks may still be irrelevant to the query."
    },
    {
        "id": 11,
        "score": 0.61,
        "text": "Context window size is a hard constraint in most LLMs."
    },
    {
        "id": 12,
        "score": 0.55,
        "text": "Diversity in retrieved chunks can improve answer quality."
    }
]
class Retriever:
    def __init__(self, chunks_list):
        self.chunks = chunks_list

    def retrieve(self, query: str = "", k: int = 10):
        """Always returns exactly k chunks by cycling through available ones."""
        if not self.chunks:
            return []

        n = len(self.chunks)
        #choose the first k relements
        #makes a copy of the dictionary->prevents modification

        return [self.chunks[i % n].copy() for i in range(k)]
        #self.chunks[] is the point where we get the chunk at the
        #wrapped index
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


retriever = Retriever(chunks)
assembler = ContextAssembler()

retrieved_chunks = retriever.retrieve(k=10)
final_context = assembler.assemble(retrieved_chunks)

print(f"Received {len(retrieved_chunks)} chunks from the retrieval")
print(f"Selected {len(final_context)} chunks for the model\n")

for i, chunk in enumerate(final_context,1):
    print(f"{i}.[scores={chunk['score']}] {chunk['text']}")
