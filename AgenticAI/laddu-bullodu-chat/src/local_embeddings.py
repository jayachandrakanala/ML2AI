"""Local Sentence Transformers adapter for the EmbeddingGemma model."""

import threading

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


_model_lock = threading.Lock()


class LocalEmbeddingGemmaEmbeddings(Embeddings):
    """Generate EmbeddingGemma vectors locally with Sentence Transformers."""

    MODEL_NAME = "unsloth/embeddinggemma-300m"

    def __init__(self) -> None:
        """Load the shared CPU SentenceTransformer model under a process lock."""
        with _model_lock:
            self.model = SentenceTransformer(self.MODEL_NAME, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode document passages with the model's ``passage:`` task prefix.

        Returned vectors are L2-normalized and converted to regular Python lists
        to satisfy LangChain's ``Embeddings`` interface.
        """
        prefixed_texts = [f"passage: {text}" for text in texts]
        vectors = self.model.encode(
            prefixed_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encode one retrieval query with the model's ``query:`` task prefix.

        The vector is L2-normalized and returned as a Python list, matching the
        document-embedding representation used by the local Chroma collection.
        """
        vector = self.model.encode(
            f"query: {text}",
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()