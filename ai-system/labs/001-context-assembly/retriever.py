from sentence_transformers import SentenceTransformer, util

class SemanticRetriever:
    def __init__(self, chunks):
        """
        Initializes the semantic retriever with a corpus of chunks.
        
        Args:
            chunks: List of dictionaries representing document chunks.
        """
        self.chunks = chunks
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        chunk_texts = [chunk["text"] for chunk in chunks]
        self.chunk_embeddings = self.model.encode(chunk_texts)

    def retrieve(self, query: str = "", k: int = 10):
        """
        Retrieves top k chunks matching the query using cosine similarity.
        """
        if not self.chunks:
            return []
        
        query_embedding = self.model.encode(query)
        cos_scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]

        scored_chunks = []
        for idx, score in enumerate(cos_scores):
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            scored_chunks.append(chunk)

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:k]
