from functools import lru_cache
from typing import cast

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.constants import CHUNK_OVERLAP, CHUNK_SIZE


@lru_cache(maxsize=4)
def _get_splitter(
    chunk_size: int, chunk_overlap: int
) -> RecursiveCharacterTextSplitter:

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    return cast(
        list[str], _get_splitter(chunk_size, chunk_overlap).split_text(text)
    )
