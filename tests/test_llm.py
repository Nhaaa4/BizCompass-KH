from typing import cast

import pytest
from langchain_core.messages import AIMessage

from bizcompass_kh.config.settings import Settings
from bizcompass_kh.services import llm
from bizcompass_kh.services.llm import GeminiProvider, OpenAIProvider, create_provider


def settings(**updates: object) -> Settings:
    return Settings.model_validate(
        {
            "llm_model": "configured-model",
            "gemini_api_key": "secret",
            **updates,
        }
    )


def test_provider_factory_defaults_to_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bizcompass_kh.services.llm.ChatGoogleGenerativeAI", lambda **_kwargs: object()
    )
    provider = create_provider(settings())
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "configured-model"


def test_provider_factory_selects_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bizcompass_kh.services.llm.ChatGoogleGenerativeAI", lambda **_kwargs: object()
    )
    provider = create_provider(settings(gemini_api_key="secret"))
    assert isinstance(provider, GeminiProvider)


def test_gemini_requires_api_key() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        settings(gemini_api_key=None)


def test_provider_factory_selects_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bizcompass_kh.services.llm.ChatOpenAI", lambda **_kwargs: object())

    provider = create_provider(
        settings(llm_provider="openai", gemini_api_key=None, openai_api_key="secret")
    )

    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "configured-model"


def test_gemini_parses_langchain_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    message = AIMessage(
        content="Grounded",
        usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    )

    class FakeChat:
        def invoke(self, _prompt: str) -> AIMessage:
            return message

    monkeypatch.setattr(
        "bizcompass_kh.services.llm.ChatGoogleGenerativeAI", lambda **_kwargs: FakeChat()
    )
    provider = GeminiProvider("model", "key")
    response = provider.generate("prompt")
    assert response.text == "Grounded"
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 2
    assert cast(object, provider.client) is not None


def test_gemini_extracts_text_from_langchain_content_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = AIMessage(
        content=[
            {
                "type": "text",
                "text": (
                    '{"relevance":5,"groundedness":5,"completeness":5,'
                    '"citation_correctness":5,"unsupported_claims":[]}'
                ),
            }
        ]
    )

    class FakeChat:
        def invoke(self, _prompt: str) -> AIMessage:
            return message

    monkeypatch.setattr(
        "bizcompass_kh.services.llm.ChatGoogleGenerativeAI", lambda **_kwargs: FakeChat()
    )
    response = GeminiProvider("model", "key").generate("prompt")

    assert response.text.startswith('{"relevance":5')
