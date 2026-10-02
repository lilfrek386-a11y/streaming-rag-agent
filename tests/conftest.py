import io
import os
from typing import cast

import numpy as np
import pytest
from rich.console import Console

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("GROQ_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("GROQ_MODEL", "test-model")
os.environ.setdefault("EMBEDDER_MODEL_NAME", "test-embedder")
os.environ.setdefault("CHAT_LOG_DIR", "/tmp/ai-internship-tests/logs")


@pytest.fixture
def console() -> Console:
    return Console(file=io.StringIO(), width=100, force_terminal=False)


class FakeEmbedder:
    VOCAB = ["cat", "dog", "python", "faiss", "rag", "audio", "test", "chunk"]

    def embed(self, text: str) -> np.ndarray:
        lowered = text.lower()
        vec = np.array(
            [float(lowered.count(word)) for word in self.VOCAB], dtype="float32"
        )
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec[0] = 1.0
            norm = 1.0
        return cast(np.ndarray, vec / norm)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()
