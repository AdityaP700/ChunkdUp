# ai-system/chunkdup/embeddings.py
from typing import List, Optional
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for semantic search."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name: SentenceTransformer model name
                       Options: all-MiniLM-L6-v2 (384 dims, fast)
                                all-mpnet-base-v2 (768 dims, more accurate)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Lazy load the model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def generate(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a text string.

        Args:
            text: Text to embed

        Returns:
            List of floats (embedding vector) or None if failed
        """
        if not text or not text.strip():
            return None

        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def generate_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [None] * len(texts)

    @property
    def embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()