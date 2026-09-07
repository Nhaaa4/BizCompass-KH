import hashlib
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import dlt
import httpx
import yaml
from sqlalchemy import delete
from sqlalchemy.orm import Session

from bizcompass_kh.config.settings import Settings, get_settings
from bizcompass_kh.databases.models import Chunk, IngestionFailure, SourceDocument
from bizcompass_kh.databases.postgres import Postgres
from bizcompass_kh.schemas.documents import SourceConfig
from bizcompass_kh.services.chunking import ChunkingService
from bizcompass_kh.services.document_processor import DocumentProcessor
from bizcompass_kh.services.embedding import EmbeddingService, create_embedding_service


@dataclass
class IngestionResult:
    source_id: str
    status: str
    chunks: int = 0
    error: str | None = None


ProgressReporter = Callable[[SourceConfig, IngestionResult, int, int], None]


def load_sources(path: str | Path) -> list[SourceConfig]:
    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, list):
        raise ValueError("sources YAML must contain a list")
    return [SourceConfig.model_validate(item) for item in payload]


def content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def stable_chunk_id(source_id: str, chunk_index: int, content: str) -> str:
    value = f"{source_id}:{chunk_index}:{content}".encode()
    return hashlib.sha256(value).hexdigest()


class IngestionService:
    def __init__(
        self,
        processor: DocumentProcessor,
        chunker: ChunkingService,
        embedder: EmbeddingService,
        embedding_fingerprint: str,
    ) -> None:
        self.processor = processor
        self.chunker = chunker
        self.embedder = embedder
        self.embedding_fingerprint = embedding_fingerprint

    def ingest_source(self, session: Session, source: SourceConfig) -> IngestionResult:
        raw = self.processor.download(source)
        digest = content_hash(raw)
        existing = session.get(SourceDocument, source.id)
        if (
            existing
            and existing.content_hash == digest
            and existing.status == "ready"
            and existing.embedding_fingerprint == self.embedding_fingerprint
        ):
            return IngestionResult(source.id, "skipped", len(existing.chunks))

        pages = (
            self.processor.process_html(raw.decode("utf-8", errors="replace"), source)
            if source.type == "html"
            else self.processor.process_pdf(raw, source)
        )
        chunks = self.chunker.chunk_documents(pages)
        if not chunks:
            raise ValueError("document contains no extractable text")
        vectors = self.embedder.embed_documents([chunk.page_content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedding count does not match chunk count")

        document = existing or SourceDocument(source_id=source.id)
        document.title = source.title
        document.agency = source.agency
        document.url = str(source.url)
        document.topic = source.topic
        document.category = source.category
        document.published_year = source.published_year
        document.official = source.official
        document.source_type = source.type
        document.content_hash = digest
        document.embedding_fingerprint = self.embedding_fingerprint
        document.status = "ready"
        document.error = None
        session.add(document)
        session.flush()
        session.execute(delete(Chunk).where(Chunk.source_id == source.id))
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            session.add(
                Chunk(
                    chunk_id=stable_chunk_id(source.id, index, chunk.page_content),
                    source_id=source.id,
                    chunk_index=index,
                    page=cast(int | None, chunk.metadata.get("page")),
                    content=chunk.page_content,
                    token_count=int(chunk.metadata["token_count"]),
                    embedding=vector,
                )
            )
        return IngestionResult(source.id, "processed", len(chunks))

    def run(
        self,
        database: Postgres,
        sources: list[SourceConfig],
        on_progress: ProgressReporter | None = None,
    ) -> list[IngestionResult]:
        results: list[IngestionResult] = []
        total = len(sources)
        for completed, source in enumerate(sources, start=1):
            try:
                with database.session() as session:
                    result = self.ingest_source(session, source)
            except Exception as error:
                message = str(error)
                with database.session() as session:
                    session.add(IngestionFailure(source_id=source.id, error=message))
                result = IngestionResult(source.id, "failed", error=message)
            results.append(result)
            if on_progress:
                on_progress(source, result, completed, total)
        return results


@dlt.resource(name="ingestion_runs", write_disposition="append")
def ingestion_runs(results: list[IngestionResult]) -> Iterator[dict[str, Any]]:
    for result in results:
        yield asdict(result)


def format_progress(
    source: SourceConfig, result: IngestionResult, completed: int, total: int
) -> str:
    percentage = completed / total * 100 if total else 100.0
    error = f" - {result.error}" if result.error else ""
    return (
        f"[{completed}/{total} | {percentage:5.1f}%] {source.title}: "
        f"{result.status} ({result.chunks} chunks){error}"
    )


def run_pipeline(
    settings: Settings | None = None, on_progress: ProgressReporter | None = None
) -> list[IngestionResult]:
    config = settings or get_settings()
    embedder = create_embedding_service(config)
    embedder.warmup()
    database = Postgres(config.postgres_dsn, config.embedding_dimension)
    database.init_database()
    with httpx.Client() as client:
        service = IngestionService(
            DocumentProcessor(http_client=client, timeout_seconds=60),
            ChunkingService(
                config.chunk_min_tokens,
                config.chunk_max_tokens,
                config.chunk_overlap_tokens,
            ),
            embedder,
            config.embedding_fingerprint,
        )
        results = service.run(database, load_sources(config.sources_path), on_progress)

    pipeline = dlt.pipeline(
        pipeline_name="bizcompass_ingestion",
        destination=dlt.destinations.postgres(credentials=config.postgres_dsn),
        dataset_name="pipeline_metadata",
    )
    pipeline.run(ingestion_runs(results))
    database.close()
    return results


def main() -> None:
    def print_progress(
        source: SourceConfig, result: IngestionResult, completed: int, total: int
    ) -> None:
        print(format_progress(source, result, completed, total), flush=True)

    run_pipeline(on_progress=print_progress)


if __name__ == "__main__":
    main()
