import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai-system'))

from chunkdup import Memory

memory = Memory()
memory.remember("I use Python")
print("[+] Works!")