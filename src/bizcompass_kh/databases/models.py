from datetime import datetime

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SourceDocument(Base):
    __tablename__ = "documents"

    source_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    agency: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    topic: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(255))
    published_year: Mapped[int | None] = mapped_column(Integer)
    official: Mapped[bool] = mapped_column(Boolean, default=False)
    source_type: Mapped[str] = mapped_column(String(20))
    content_hash: Mapped[str] = mapped_column(String(64))
    embedding_fingerprint: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(20), default="ready")
    error: Mapped[str | None] = mapped_column(Text)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("source_id", "chunk_index"),
        Index("chunks_source_id_idx", "source_id"),
        Index(
            "chunks_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("documents.source_id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(VECTOR())
    document: Mapped[SourceDocument] = relationship(back_populates="chunks")


def configure_embedding_dimension(dimension: int) -> None:
    """Configure the pgvector type before SQLAlchemy creates the chunks table."""
    if dimension < 1:
        raise ValueError("embedding dimension must be positive")
    Chunk.__table__.c.embedding.type = VECTOR(dimension)


class IngestionFailure(Base):
    __tablename__ = "ingestion_failures"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(String(255), index=True)
    error: Mapped[str] = mapped_column(Text)
    failed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RagRequest(Base):
    __tablename__ = "rag_requests"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(String(20))
    model: Mapped[str] = mapped_column(String(255))
    latency_ms: Mapped[float] = mapped_column(Float)
    retrieved_chunks: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RagRetrieval(Base):
    __tablename__ = "rag_retrievals"

    request_id: Mapped[int] = mapped_column(
        ForeignKey("rag_requests.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[int] = mapped_column(
        ForeignKey("rag_requests.id", ondelete="CASCADE"), index=True
    )
    score: Mapped[int] = mapped_column(SmallInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvaluationScore(Base):
    __tablename__ = "evaluation_scores"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    question_id: Mapped[str] = mapped_column(String(64))
    relevance: Mapped[int] = mapped_column(SmallInteger)
    groundedness: Mapped[int] = mapped_column(SmallInteger)
    completeness: Mapped[int] = mapped_column(SmallInteger)
    citation_correctness: Mapped[int] = mapped_column(SmallInteger)
    hallucinated: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
