from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from bizcompass_kh.databases.models import Base, configure_embedding_dimension


class Postgres:
    def __init__(self, database_url: str, embedding_dimension: int) -> None:
        self.embedding_dimension = embedding_dimension
        configure_embedding_dimension(embedding_dimension)
        self.engine: Engine = create_engine(database_url, pool_pre_ping=True)
        self._session_factory = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._session_factory.begin() as session:
            yield session

    def ping(self) -> None:
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def init_database(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            current_dimension = connection.execute(
                text(
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = to_regclass('public.chunks') AND attname = 'embedding'"
                )
            ).scalar_one_or_none()
            if current_dimension == -1:
                mismatched_vectors = connection.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM chunks "
                        "WHERE vector_dims(embedding) <> :dimension)"
                    ),
                    {"dimension": self.embedding_dimension},
                ).scalar_one()
                if mismatched_vectors:
                    raise RuntimeError(
                        "Existing chunks use a different embedding dimension. "
                        "Re-ingest them with the configured embedding model before creating the "
                        "index."
                    )
                connection.execute(
                    text(
                        "ALTER TABLE chunks ALTER COLUMN embedding TYPE "
                        f"vector({self.embedding_dimension}) USING "
                        f"embedding::vector({self.embedding_dimension})"
                    )
                )
            elif current_dimension is not None and current_dimension != self.embedding_dimension:
                raise RuntimeError(
                    "Existing chunks use a different embedding dimension. "
                    "Re-ingest them with the configured embedding model before creating the index."
                )
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS "
                    "embedding_fingerprint VARCHAR(512)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks "
                    "USING GIN (to_tsvector('english', content))"
                )
            )

    def close(self) -> None:
        self.engine.dispose()
