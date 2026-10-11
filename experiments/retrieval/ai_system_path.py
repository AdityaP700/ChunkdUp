# experiments/retrieval/ai_system_path.py
"""Puts ai-system/ on sys.path so experiments can import the chunkdup package."""
import sys
from pathlib import Path

AI_SYSTEM_DIR = str(Path(__file__).resolve().parent.parent.parent / "ai-system")
if AI_SYSTEM_DIR not in sys.path:
    sys.path.insert(0, AI_SYSTEM_DIR)


def ai_system_path():
    return AI_SYSTEM_DIR
