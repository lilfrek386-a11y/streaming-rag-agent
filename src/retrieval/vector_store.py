import faiss

from src.retrieval.embedder import Embedder


class VectorStore:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._index: faiss.IndexFlatIP | None = None
        self._entries: list[dict] = []

    def add_text(self, text: str, source: str) -> str:
        doc_id = str(len(self._entries))
        vector = self._embedder.embed(text).astype("float32")

        if self._index is None:
            self._index = faiss.IndexFlatIP(vector.shape[0])

        self._index.add(vector.reshape(1, -1))
        self._entries.append({"id": doc_id, "text": text, "source": source})
        return doc_id

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

    def search(self, query: str, top_n: int = 3) -> list[dict]:
        if self._index is None or self._index.ntotal == 0:
            return []

        query_vector = self._embedder.embed(query).astype("float32").reshape(1, -1)
        distances, indices = self._index.search(
            query_vector, min(top_n, self._index.ntotal)
        )

        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            entry = self._entries[idx]
            results.append(
                {"text": entry["text"], "source": entry["source"], "score": float(dist)}
            )

        return results

    def get_by_id(self, doc_id: str) -> dict | None:
        idx = int(doc_id)
        if 0 <= idx < len(self._entries):
            return self._entries[idx]
        return None
