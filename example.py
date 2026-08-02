from chunkdup import Memory

# Initialize with auto-discovery of labs directory
memory = Memory(data_dir="./data", llm_provider="gemini")

# Add conversation
memory.add("I'm building ChunkdUp in Python")
memory.add("I prefer Neovim as my editor")

# Query with LLM
result = memory.query("What project am I building?", use_llm=True)
print(result["answer"])

# Query without LLM (fallback)
result = memory.query("Which editor do I use?", use_llm=False)
print(result["answer"])