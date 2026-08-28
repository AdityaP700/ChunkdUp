#!/usr/bin/env python
"""Quickstart example for ChunkdUp."""

import sys
import os

# Add ai-system to sys.path so example works directly from source repository
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ai-system"))

from chunkdup import Memory


def main():
    # Initialize with in-memory store (no database setup required)
    memory = Memory(store="memory")

    # Remember things
    print("[+] Remembering...")
    memory.remember("I'm building ChunkdUp in Python")
    memory.remember("I use Python")
    memory.remember("My favorite editor is Neovim.")

    # Retrieve memories
    print("\n[?] Retrieving...")
    results = memory.retrieve("What language?")
    for r in results:
        print(f"  {r.get('key')}: {r.get('value')}")

    # Search
    print("\n[*] Searching...")
    results = memory.search("editor")
    for r in results:
        print(f"  {r.get('key')}: {r.get('value')}")

    # Get all memories
    print("\n[=] All memories:")
    for m in memory.get_all():
        print(f"  {m.get('key')}: {m.get('value')}")

    # Statistics
    print(f"\n[#] Stats: {memory.get_stats()}")


if __name__ == "__main__":
    main()
