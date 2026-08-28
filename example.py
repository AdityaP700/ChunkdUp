from chunkdup import Memory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai-system'))

memory = Memory()
memory.add("I use Python")
print("✅ Works!")