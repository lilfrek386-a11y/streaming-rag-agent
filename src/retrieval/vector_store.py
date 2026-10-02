import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import faiss

from src.core.constants import SEARCH_CANDIDATE_MULTIPLIER

if TYPE_CHECKING:
    from src.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.faiss"
ENTRIES_FILENAME = "entries.json"


class VectorStore:
    def __init__(self, embedder: "Embedder"):
        self._embedder = embedder
        self._index: faiss.Index | None = None
        self._entries: list[dict] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add_many(self, entries: list[dict]) -> None:
        if not entries:
            return

        texts = [e["text"] for e in entries]
        vectors = self._embedder.embed_batch(texts).astype("float32")

        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])

        self._index.add(vectors)

        for entry in entries:
            doc_id = str(len(self._entries))
            self._entries.append(
                {"id": doc_id, "text": entry["text"], "source": entry["source"]}
            )

        logger.info("indexed %d chunks (total %d)", len(entries), len(self._entries))

    def search(
        self,
        query: str,
        top_n: int = 3,
        multiplier: int = SEARCH_CANDIDATE_MULTIPLIER,
        max_per_source: int = 3,
    ) -> list[dict]:
        top_n = max(1, top_n)
        if self._index is None or self._index.ntotal == 0:
            return []

        query_vector = self._embedder.embed(query).astype("float32").reshape(1, -1)
        distances, indices = self._index.search(
            query_vector, min(top_n * multiplier, self._index.ntotal)
        )

        results: list[dict[str, Any]] = []
        source_count: dict[str, int] = {}

        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx == -1:
                continue

            entry = self._entries[idx]
            source = entry["source"]
            current_count = source_count.get(source, 0)

            if current_count >= max_per_source:
                continue

            results.append(
                {"text": entry["text"], "source": source, "score": float(dist)}
            )

            source_count[source] = current_count + 1

            if len(results) == top_n:
                break

        logger.debug("search %r -> %d results", query[:60], len(results))
        return results

    def save(self, directory: Path, meta: dict) -> None:
        if self._index is None or not self._entries:
            return

        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / INDEX_FILENAME))
        (directory / ENTRIES_FILENAME).write_text(
            json.dumps({"meta": meta, "entries": self._entries}, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("saved index with %d chunks to %s", len(self._entries), directory)

    def load(self, directory: Path, expected_meta: dict) -> bool:
        index_path = directory / INDEX_FILENAME
        entries_path = directory / ENTRIES_FILENAME

        if not index_path.is_file() or not entries_path.is_file():
            return False

        try:
            payload = json.loads(entries_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("unreadable index cache at %s, rebuilding", directory)
            return False

        if payload.get("meta") != expected_meta:
            logger.info("index cache is stale, rebuilding")
            return False

        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            return False

        try:
            index = faiss.read_index(str(index_path))
        except Exception:  # noqa: BLE001
            logger.warning("corrupt FAISS index at %s, rebuilding", index_path)
            return False

        if index.ntotal != len(entries):
            logger.warning("index/entries length mismatch, rebuilding")
            return False

        self._index = index
        self._entries = entries
        logger.info("loaded %d chunks from cache", len(entries))
        return True
