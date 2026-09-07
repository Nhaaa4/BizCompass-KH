from dataclasses import dataclass
from typing import Protocol

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from bizcompass_kh.config.settings import ProviderName, Settings


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMProvider(Protocol):
    provider_name: str
    model: str

    def generate(self, prompt: str) -> LLMResponse: ...


def extract_message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_blocks = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        if text_blocks:
            return "\n".join(text_blocks)
    return str(content)


class GeminiProvider:
    provider_name = "gemini"

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.client = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

    def generate(self, prompt: str) -> LLMResponse:
        message = self.client.invoke(prompt)
        usage = message.usage_metadata or {}
        text = extract_message_text(message.content)
        return LLMResponse(
            text=text.strip(),
            prompt_tokens=int(usage.get("input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
        )


class OpenAIProvider:
    provider_name = "openai"

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.client = ChatOpenAI(model=model, api_key=api_key)

    def generate(self, prompt: str) -> LLMResponse:
        message = self.client.invoke(prompt)
        usage = message.usage_metadata or {}
        return LLMResponse(
            text=extract_message_text(message.content).strip(),
            prompt_tokens=int(usage.get("input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
        )


def create_provider(
    settings: Settings,
    *,
    provider: ProviderName | None = None,
    model: str | None = None,
) -> LLMProvider:
    selected_provider = provider or settings.llm_provider
    selected_model = model or settings.llm_model
    if selected_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set for OpenAI")
        return OpenAIProvider(selected_model, settings.openai_api_key)
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY must be set for Gemini")
    return GeminiProvider(selected_model, settings.gemini_api_key)
