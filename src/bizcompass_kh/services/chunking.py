import re
from collections.abc import Sequence

from langchain_core.documents import Document

_TOKEN_PATTERN = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    """Count whitespace-delimited tokens; deterministic for chunk bounds and tests."""
    return len(_TOKEN_PATTERN.findall(text))


class ChunkingService:
    def __init__(
        self,
        min_tokens: int = 500,
        max_tokens: int = 700,
        overlap_tokens: int = 50,
    ) -> None:
        if not 0 <= overlap_tokens < min_tokens <= max_tokens:
            raise ValueError("expected 0 <= overlap_tokens < min_tokens <= max_tokens")
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_documents(self, documents: Sequence[Document]) -> list[Document]:
        chunks: list[Document] = []
        chunk_index = 0
        for document in documents:
            words = document.page_content.split()
            if not words:
                continue
            for start, end in self._bounds(len(words)):
                metadata = dict(document.metadata)
                metadata["chunk_index"] = chunk_index
                metadata["token_count"] = end - start
                chunks.append(Document(page_content=" ".join(words[start:end]), metadata=metadata))
                chunk_index += 1
        return chunks

    def _bounds(self, total: int) -> list[tuple[int, int]]:
        if total <= self.max_tokens:
            return [(0, total)]

        starts = list(range(0, total, self.max_tokens - self.overlap_tokens))
        bounds: list[tuple[int, int]] = []
        for start in starts:
            end = min(start + self.max_tokens, total)
            if end - start < self.min_tokens and bounds:
                previous_start, _ = bounds[-1]
                bounds[-1] = (previous_start, total)
                break
            bounds.append((start, end))
            if end == total:
                break
        return bounds
