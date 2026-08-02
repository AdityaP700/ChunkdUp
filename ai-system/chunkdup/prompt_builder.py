class PromptBuilder:
    def build(self, query: str, contexts: dict, variant: str = "expert") -> str:

        memories = contexts.get("memories", [])
        #documents are the retrieved memories
        #later these could be used to retrieve from massive chunks from the vector database
        documents = contexts.get("documents", [])
        #currently we have created the socket ,whenever we are ready to connect the lab 1 with lab4 then we can easily do

        # Build memories block
        if not memories:
            memories_str = "No relevant memories found."
        else:
            memory_lines = []
            for i, mem in enumerate(memories, 1):
                # Extract key/value pair nicely
                k = mem.get("key", "unknown")
                v = mem.get("value", "unknown")
                memory_lines.append(f"{i}. {k.capitalize()} = {v}")
            memories_str = "\n".join(memory_lines)

        # Build documents block (if any exist)
        if not documents:
            documents_str = "No relevant documents found."
        else:
            doc_lines = []
            # We offset the citation numbers so they don't clash with memories
            offset = len(memories)
            for i, doc in enumerate(documents, 1):
                text = doc.get("text", "").strip()
                doc_lines.append(f"{i + offset}. {text}")
            documents_str = "\n".join(doc_lines)
#we are creating the prompts while taking the retrieved memories
#those retrieved memories will be framed in the form of a prompt
# which later will be injected into the LLM for final result
        if variant == "expert":
            return f"""You are an expert AI assistant.

Use ONLY the provided memories and documents.
If the answer cannot be inferred, the "answer" field should be "I don't know."
Always be concise and accurate.

You MUST respond ONLY with a valid JSON object in the following format:
{{
  "answer": "your answer here",
  "confidence": <float between 0.0 and 1.0>,
  "citations": <list of integer citation numbers used to answer the question>
}}

Relevant Memories:
----------------
{memories_str}
----------------

Retrieved Documents:
----------------
{documents_str}
----------------

Question:
{query}

Answer:"""
        else:
            # Fallback to expert if basic isn't defined
            return self.build(query, contexts, "expert")
