from pathlib import Path

from src.retrieval.vector_store import VectorStore


def _store(embedder) -> VectorStore:
    store = VectorStore(embedder)
    store.add_many(
        [
            {"text": "cat cat cat", "source": "a.txt"},
            {"text": "cat cat dog", "source": "a.txt"},
            {"text": "cat dog dog", "source": "a.txt"},
            {"text": "cat python", "source": "b.txt"},
        ]
    )
    return store


def test_search_on_empty_store_returns_nothing(embedder):
    assert VectorStore(embedder).search("cat") == []


def test_search_returns_scored_results(embedder):
    results = _store(embedder).search("cat", top_n=2)

    assert len(results) == 2
    assert results[0]["source"] in {"a.txt", "b.txt"}
    assert results[0]["score"] >= results[1]["score"]


def test_diversity_reranking_caps_one_source(embedder):
    results = _store(embedder).search("cat", top_n=3, max_per_source=2)

    sources = [r["source"] for r in results]
    assert sources.count("a.txt") <= 2
    assert "b.txt" in sources


def test_entry_ids_stay_aligned_with_index(embedder):
    store = _store(embedder)
    assert len(store) == 4
    assert [e["id"] for e in store._entries] == ["0", "1", "2", "3"]


def test_save_and_load_roundtrip(embedder, tmp_path: Path):
    meta = {"embedder": "test", "corpus": []}
    _store(embedder).save(tmp_path, meta)

    restored = VectorStore(embedder)
    assert restored.load(tmp_path, expected_meta=meta) is True
    assert len(restored) == 4
    assert restored.search("python", top_n=1)[0]["source"] == "b.txt"


def test_load_rejects_stale_meta(embedder, tmp_path: Path):
    _store(embedder).save(tmp_path, {"embedder": "old-model", "corpus": []})

    restored = VectorStore(embedder)
    assert (
        restored.load(tmp_path, expected_meta={"embedder": "new-model", "corpus": []})
        is False
    )
    assert len(restored) == 0


def test_load_without_cache_returns_false(embedder, tmp_path: Path):
    assert VectorStore(embedder).load(tmp_path / "missing", expected_meta={}) is False


def test_load_survives_a_corrupt_index(embedder, tmp_path: Path):
    meta = {"embedder": "test", "corpus": []}
    _store(embedder).save(tmp_path, meta)
    (tmp_path / "index.faiss").write_bytes(b"not an index")

    assert VectorStore(embedder).load(tmp_path, expected_meta=meta) is False


def test_empty_store_is_not_saved(embedder, tmp_path: Path):
    VectorStore(embedder).save(tmp_path, {})
    assert not (tmp_path / "index.faiss").exists()
