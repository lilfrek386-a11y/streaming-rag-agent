import logging
from typing import cast

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.settings import settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str | None = None):
        resolved = model_name or settings.embedder.model_name
        logger.info("loading embedding model %s", resolved)
        self._model = SentenceTransformer(resolved)

    def embed(self, text: str) -> np.ndarray:
        return cast(
            np.ndarray,
            self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True),
        )

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return cast(
            np.ndarray,
            self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True),
        )
