import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.settings import settings


class Embedder:
    def __init__(self, model_name: str = settings.embedder.model_name):
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
