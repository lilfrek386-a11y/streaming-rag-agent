import re
from pathlib import Path

from src.retrieval.chunker import chunk_text
from src.core.constants import CHUNK_SIZE, CHUNK_OVERLAP


def load_corpus(
    corpus_dir: Path, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[dict]:

    entries: list[dict] = []

    for file_path in sorted(corpus_dir.glob("*.txt")):
        raw_text = file_path.read_text(encoding="utf-8").strip()

        if not raw_text:
            continue

        chunks = chunk_text(
            raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for chunk in chunks:
            cleaned = clean_extracted_text(chunk)
            if cleaned:
                entries.append({"text": cleaned, "source": file_path.name})

    return entries


def clean_extracted_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text)

    cleaned = re.sub(r"[\x00-\x1F\x7F]", "", cleaned)

    return cleaned.strip()
