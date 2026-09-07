from collections.abc import Sequence
from typing import Protocol

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from bizcompass_kh.config.settings import Settings


class EmbeddingService(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbeddingService:
    def __init__(self, model_name: str, api_key: str, dimension: int) -> None:
        self.model_name = model_name
        self._dimension = dimension
        self._client = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
            output_dimensionality=dimension,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def warmup(self) -> None:
        """The Gemini client is initialized eagerly; no local model weights are downloaded."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


class OpenAIEmbeddingService:
    def __init__(self, model_name: str, api_key: str, dimension: int) -> None:
        self.model_name = model_name
        self._dimension = dimension
        self._client = OpenAIEmbeddings(model=model_name, api_key=api_key, dimensions=dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def warmup(self) -> None:
        """The OpenAI client is initialized eagerly; no local model weights are downloaded."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


def create_embedding_service(settings: Settings) -> EmbeddingService:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set for OpenAI embeddings")
        return OpenAIEmbeddingService(
            settings.embedding_model, settings.openai_api_key, settings.embedding_dimension
        )
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY must be set for Gemini embeddings")
    return GeminiEmbeddingService(
        settings.embedding_model, settings.gemini_api_key, settings.embedding_dimension
    )
