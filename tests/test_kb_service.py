from pathlib import Path

import pytest

from src.retrieval.kb_service import KnowledgeBaseService
from src.retrieval.vector_store import VectorStore


def _encoder_available() -> bool:
    try:
        import tiktoken

        tiktoken.get_encoding("cl100k_base")
        return True
    except Exception:  # noqa: BLE001
        return False


needs_encoder = pytest.mark.skipif(
    not _encoder_available(), reason="cl100k_base encoding unavailable offline"
)


def test_blank_document_is_ignored(embedder):
    service = KnowledgeBaseService(VectorStore(embedder))

    assert service.add_document("   \n  ", source="x") == 0


@needs_encoder
def test_document_is_chunked_and_cleaned(embedder):
    store = VectorStore(embedder)
    service = KnowledgeBaseService(store)

    count = service.add_document("cat\x00 sat\n\n on   the mat", source="a.txt")

    assert count == 1
    assert len(store) == 1
    assert "\x00" not in store._entries[0]["text"]
    assert "  " not in store._entries[0]["text"]


@needs_encoder
def test_corpus_directory_is_loaded(embedder, tmp_path: Path):
    (tmp_path / "a.txt").write_text("cat dog", encoding="utf-8")
    (tmp_path / "b.txt").write_text("python faiss", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("not indexed", encoding="utf-8")
    store = VectorStore(embedder)

    total = KnowledgeBaseService(store).load_corpus_from_dir(tmp_path)

    assert total == 2
    assert {e["source"] for e in store._entries} == {"a.txt", "b.txt"}


def test_cache_meta_changes_when_corpus_changes(tmp_path: Path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("cat", encoding="utf-8")

    before = KnowledgeBaseService.build_cache_meta(tmp_path)
    file_path.write_text("cat dog elephant", encoding="utf-8")
    after = KnowledgeBaseService.build_cache_meta(tmp_path)

    assert before != after


def test_cache_meta_on_missing_directory(tmp_path: Path):
    meta = KnowledgeBaseService.build_cache_meta(tmp_path / "nope")

    assert meta["corpus"] == []
