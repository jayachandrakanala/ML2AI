"""Local Sentence Transformers adapter for the EmbeddingGemma model."""

import threading

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


_model_lock = threading.Lock()


class LocalEmbeddingGemmaEmbeddings(Embeddings):
    """Generate EmbeddingGemma vectors locally with Sentence Transformers."""

    MODEL_NAME = "unsloth/embeddinggemma-300m"

    def __init__(self) -> None:
        with _model_lock:
            self.model = SentenceTransformer(self.MODEL_NAME, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document passages using the model's passage task prefix."""
        prefixed_texts = [f"passage: {text}" for text in texts]
        vectors = self.model.encode(
            prefixed_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query using the model's query task prefix."""
        vector = self.model.encode(
            f"query: {text}",
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()