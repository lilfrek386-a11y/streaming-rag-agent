import logging
import re
from pathlib import Path

from src.core.constants import CHUNK_OVERLAP, CHUNK_SIZE
from src.core.settings import settings
from src.retrieval.chunker import chunk_text
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store

    def add_document(self, text: str, source: str) -> int:
        if not text.strip():
            return 0

        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        entries = []

        for chunk in chunks:
            cleaned = self._clean_extracted_text(chunk)
            if cleaned:
                entries.append({"text": cleaned, "source": source})

        if entries:
            self._vector_store.add_many(entries)

        return len(entries)

    def load_corpus_from_dir(self, corpus_dir: Path) -> int:
        total_chunks = 0
        for file_path in sorted(corpus_dir.glob("*.txt")):
            try:
                raw_text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                logger.exception("skipping unreadable corpus file %s", file_path)
                continue
            total_chunks += self.add_document(raw_text, source=file_path.name)

        return total_chunks

    @staticmethod
    def build_cache_meta(corpus_dir: Path) -> dict:
        corpus = []
        if corpus_dir.is_dir():
            for file_path in sorted(corpus_dir.glob("*.txt")):
                stat = file_path.stat()
                corpus.append(
                    {
                        "name": file_path.name,
                        "size": stat.st_size,
                        "mtime": int(stat.st_mtime),
                    }
                )

        return {
            "embedder": settings.embedder.model_name,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "corpus": corpus,
        }

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text)
        cleaned = re.sub(r"[\x00-\x1F\x7F]", "", cleaned)
        return cleaned.strip()
