# class Retriever:
#     def __init__(self, chunks_list):
#         self.chunks = chunks_list

#     #measures how relevant a chunk is to the query by counting how many words
#     #in common between them
#     def _calculate_overlap(self, query: str,chunk_keywords:list)->int:
#         query_words=set(query.lower().split())
#         chunk_words=set(word.lower() for word in chunk_keywords)
#         return len(query_words & chunk_words)

#     #i call retriever.retrieve(query,k)
#     #it creates an empty list called scored_chunks=[]
#     def retrieve(self, query: str = "", k: int = 10):
#         """Always returns exactly k chunks by cycling through available ones."""
#         if not self.chunks:
#             return []
#         scored_chunks=[]
#         #for each chunk,it calls the _calculate...it gets back a number

#         for chunk in self.chunks:
#             #it creates a tuplelate_overlap(query,chunk.get("keywords",[]))
#             #it adds this tuple to scored_chunk
#             scored_chunks.append((overlap,chunk["score"],chunk))

#         #it sorts scored_chunks
#         scored_chunks.sort(key=lambda x:(x[0],x[1]),reverse=True)
#         #it takes the top k chunks from the sorted list
#         top_chunks=[chunk.copy() for _, _, chunk in scored_chunks[:k]]
#         return top_chunks (overlap,original_score,chunk)
