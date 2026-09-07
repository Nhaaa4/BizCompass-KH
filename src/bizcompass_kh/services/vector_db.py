from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Select, desc, func, select
from sqlalchemy.orm import Session

from bizcompass_kh.databases.models import Chunk, SourceDocument
from bizcompass_kh.services.embedding import EmbeddingService


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    source_id: str
    content: str
    page: int | None
    title: str
    agency: str
    url: str
    official: bool
    score: float


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    document: SourceDocument
    score: float


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RankedChunk]],
    *,
    limit: int,
    rank_constant: int = 60,
) -> list[RankedChunk]:
    by_id: dict[str, RankedChunk] = {}
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            chunk_id = result.chunk.chunk_id
            by_id[chunk_id] = result
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank_constant + rank)
    ordered = sorted(
        by_id.values(),
        key=lambda item: (scores[item.chunk.chunk_id], item.document.official),
        reverse=True,
    )
    return [
        RankedChunk(item.chunk, item.document, scores[item.chunk.chunk_id])
        for item in ordered[:limit]
    ]


class VectorDBService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        *,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.embedding_service = embedding_service
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def vector_search(self, session: Session, query: str) -> list[RankedChunk]:
        vector = self.embedding_service.embed_query(query)
        distance = Chunk.embedding.cosine_distance(vector)
        statement: Select[tuple[Chunk, SourceDocument, float]] = (
            select(Chunk, SourceDocument, distance.label("distance"))
            .join(SourceDocument, Chunk.source_id == SourceDocument.source_id)
            .order_by(distance, desc(SourceDocument.official))
            .limit(self.candidate_k)
        )
        return [
            RankedChunk(chunk, document, max(0.0, 1.0 - float(distance_value)))
            for chunk, document, distance_value in session.execute(statement)
        ]

    def keyword_search(self, session: Session, query: str) -> list[RankedChunk]:
        search_query = func.websearch_to_tsquery("english", query)
        search_vector = func.to_tsvector("english", Chunk.content)
        rank = func.ts_rank_cd(search_vector, search_query)
        statement: Select[tuple[Chunk, SourceDocument, float]] = (
            select(Chunk, SourceDocument, rank.label("rank"))
            .join(SourceDocument, Chunk.source_id == SourceDocument.source_id)
            .where(search_vector.op("@@")(search_query))
            .order_by(desc(rank), desc(SourceDocument.official))
            .limit(self.candidate_k)
        )
        return [
            RankedChunk(chunk, document, float(rank_value))
            for chunk, document, rank_value in session.execute(statement)
        ]

    def hybrid_search(self, session: Session, query: str, k: int = 5) -> list[SearchResult]:
        fused = reciprocal_rank_fusion(
            [self.vector_search(session, query), self.keyword_search(session, query)],
            limit=k,
            rank_constant=self.rrf_k,
        )
        return [
            SearchResult(
                chunk_id=item.chunk.chunk_id,
                source_id=item.document.source_id,
                content=item.chunk.content,
                page=item.chunk.page,
                title=item.document.title,
                agency=item.document.agency,
                url=item.document.url,
                official=item.document.official,
                score=item.score,
            )
            for item in fused
        ]
