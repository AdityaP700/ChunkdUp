import os
import json
from sentence_transformers import SentenceTransformer, util

def load_chunks():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks_path = os.path.join(base_dir, "data", "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)

chunks = load_chunks()

class SemanticRetriever:
    def __init__(self, chunks):
        # store the chunks
        self.chunks = chunks
        # load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # Build a list containing the "text" from every chunk.
        chunk_texts=[chunk["text"] for chunk in chunks]
        self.chunk_embeddings = self.model.encode(chunk_texts)
        # print(self.chunk_embeddings.shape)


    def retrieve(self, query: str = "", k: int = 10):
        if not self.chunks:
            return []
        # encode query
        query_embedding = self.model.encode(query)
        # compare with stored embeddings
        cos_scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]

        # Zip scores with chunks
        scored_chunks = []
        for idx, score in enumerate(cos_scores):
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            scored_chunks.append(chunk)

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:k]

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
            #it creates a tuplelate_overlap(query,chunk.get("keywords",[]))
            #it adds this tuple to scored_chunk
            scored_chunks.append((overlap,chunk["score"],chunk))

        #it sorts scored_chunks
        scored_chunks.sort(key=lambda x:(x[0],x[1]),reverse=True)
        #it takes the top k chunks from the sorted list
        top_chunks=[chunk.copy() for _, _, chunk in scored_chunks[:k]]
        return top_chunks (overlap,original_score,chunk)


class ContextAssembler:
    def __init__(self):
        self.context_budget=120

#it doesnt takes any data ,it just takes the chunks
#input when assemble() is called
    def assemble(self,chunks:list[dict])->list[dict]:

        if not chunks:
            print("No chunks provided to assembler.")
            return []

        sorted_chunks=sorted(
            chunks,
            #konsa cheez ko sort out karna hai
            #kis tarike se karna hai
            #aur kya reverse hoga??nhi
            key=lambda x : x.get("score",0.0),
            reverse=True
        )

        selected = []
        current_length = 0
        budget = self.context_budget

        print(f"\n=== Context Budget: {budget} characters ===")

        for i, chunk in enumerate(sorted_chunks, 1):
            chunk_text = chunk.get("text", "")
            added_length = len(chunk_text)          # characters
            word_count = len(chunk_text.split())    # for readability

            if current_length + added_length > budget:
                print(f"✗ Skipped Chunk {i} ({word_count} words, {added_length} chars) "
                      f"→ Budget exceeded")
                continue  # V3: skip and try next

            # It fits → add it
            selected.append(chunk)
            current_length += added_length
            remaining = budget - current_length

            print(f"✓ Added Chunk {i} ({word_count} words, {added_length} chars) "
                  f"→ Remaining Budget: {remaining} chars")

        print(f"\nFinal Selection: {len(selected)} chunks "
              f"(used {current_length} / {budget} chars)\n")

        return selected

question ="what is diversity in retrieval?"

retriever = SemanticRetriever(chunks)
assembler = ContextAssembler()

retrieved_chunks = retriever.retrieve(query=question,k=10)
final_context = assembler.assemble(retrieved_chunks)

print(f"Received {len(retrieved_chunks)} chunks from the retrieval")
print(f"Selected {len(final_context)} chunks for the model\n")

for i, chunk in enumerate(final_context,1):
    print(f"{i}.[scores={chunk['score']}] {chunk['text']}")


