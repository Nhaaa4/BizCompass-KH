from collections.abc import Sequence
from pathlib import Path

import pytest

from bizcompass_kh.ingestion import dlt_pipeline
from bizcompass_kh.ingestion.dlt_pipeline import (
    IngestionResult,
    content_hash,
    load_sources,
    stable_chunk_id,
)
from bizcompass_kh.schemas.documents import SourceConfig
from bizcompass_kh.services import embedding
from bizcompass_kh.config.settings import Settings
from bizcompass_kh.services.embedding import (
    GeminiEmbeddingService,
    OpenAIEmbeddingService,
    create_embedding_service,
)


class FakeGeminiEmbeddings:
    instances: list["FakeGeminiEmbeddings"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.instances.append(self)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class FakeOpenAIEmbeddings(FakeGeminiEmbeddings):
    instances: list["FakeOpenAIEmbeddings"] = []


def test_sources_yaml_is_valid_and_nonempty() -> None:
    sources = load_sources(Path("data/sources.yaml"))
    assert len(sources) >= 10
    assert all(str(source.url).startswith("https://") for source in sources)


def test_hashes_are_stable() -> None:
    assert content_hash(b"same") == content_hash(b"same")
    assert stable_chunk_id("source", 0, "same") == stable_chunk_id("source", 0, "same")


def test_progress_format_includes_percentage_source_and_result() -> None:
    source = SourceConfig.model_validate(
        {
            "id": "registration-guide",
            "agency": "MOC",
            "title": "Business registration guide",
            "url": "https://example.com/guide.pdf",
            "topic": "registration",
            "category": "guide",
            "official": True,
            "type": "pdf",
        }
    )

    message = dlt_pipeline.format_progress(
        source, IngestionResult(source.id, "processed", 12), 2, 8
    )

    assert message == "[2/8 |  25.0%] Business registration guide: processed (12 chunks)"


def test_gemini_embedding_pipeline_uses_configured_model_and_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding, "GoogleGenerativeAIEmbeddings", FakeGeminiEmbeddings)
    service = GeminiEmbeddingService("test-model", "key", dimension=1)

    assert service.embed_documents(["one", "three"]) == [[3.0], [5.0]]
    assert service.embed_query("one") == [3.0]
    assert FakeGeminiEmbeddings.instances[-1].kwargs == {
        "model": "test-model",
        "google_api_key": "key",
        "output_dimensionality": 1,
    }


def test_embedding_factory_uses_the_selected_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding, "OpenAIEmbeddings", FakeOpenAIEmbeddings)
    config = Settings.model_validate(
        {
            "llm_provider": "openai",
            "llm_model": "gpt-5-mini",
            "openai_api_key": "key",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 768,
        }
    )

    service = create_embedding_service(config)

    assert isinstance(service, OpenAIEmbeddingService)
    assert FakeOpenAIEmbeddings.instances[-1].kwargs == {
        "model": "text-embedding-3-small",
        "api_key": "key",
        "dimensions": 768,
    }


def test_embedding_protocol_accepts_sequences() -> None:
    class Embedder:
        dimension = 1

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [float(bool(text))]

    assert Embedder().embed_documents(("a", "b")) == [[0.0], [0.0]]
