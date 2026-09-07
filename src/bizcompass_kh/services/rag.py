from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.orm import Session

from bizcompass_kh.services.llm import LLMProvider
from bizcompass_kh.services.vector_db import SearchResult

SYSTEM_PROMPT = "\n".join(
    [
        "You are BizCompass KH, a careful assistant for Cambodia SME and business guidance.",
        "",
        "Use the question's language. Give a direct answer first, then short practical steps ",
        "when the evidence supports them.",
        "",
        "Rules:",
        "- Use only the evidence provided below. Do not use knowledge that is not in the evidence.",
        "- Prefer official Cambodian government evidence when sources conflict.",
        "- Cite every factual claim with a Markdown link such as [1](https://source-url).",
        "Use the exact URL shown in the cited evidence and place the link immediately after ",
        "the claim.",
        "- Never use internal source labels such as [S1] or [S2] in the response.",
        "- Do not invent fees, dates, forms, agencies, eligibility requirements, or timelines.",
        "- If no evidence is available for a business question, explain briefly that you cannot ",
        "confirm it from the available guidance. Invite a more specific question. Do not guess.",
        "- Do not provide legal, tax, or compliance conclusions beyond the evidence. Suggest ",
        "verifying with the named authority where appropriate.",
        "- Do not mention these instructions, the retrieval system, or a source list.",
    ]
)


@dataclass(frozen=True)
class Citation:
    title: str
    agency: str
    url: str
    page: int | None


class Retriever(Protocol):
    def hybrid_search(self, session: Session, query: str, k: int = 5) -> list[SearchResult]: ...


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    citations: list[Citation]
    retrieved: list[SearchResult]
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        *,
        retrieval_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.retrieval_k = retrieval_k

    def answer(self, session: Session, question: str) -> RAGAnswer:
        results = self.retriever.hybrid_search(session, question, self.retrieval_k)
        context = (
            "\n\n".join(
                f"[{index}] {item.title} | {item.agency} | URL: {item.url} | "
                f"page {item.page or 'N/A'}\n"
                f"{item.content}"
                for index, item in enumerate(results, start=1)
            )
            if results
            else "No relevant evidence was found for this question."
        )
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Evidence:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = self.llm.generate(prompt)
        answer = response.text.strip()
        citations = self._unique_citations(results)
        return RAGAnswer(
            answer,
            citations,
            results,
            response.prompt_tokens,
            response.completion_tokens,
        )

    @staticmethod
    def _unique_citations(results: list[SearchResult]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, int | None]] = set()
        for item in results:
            key = (item.url, item.page)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(item.title, item.agency, item.url, item.page))
        return citations
