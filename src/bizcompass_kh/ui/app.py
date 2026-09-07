import re
import time
from dataclasses import dataclass
from html import escape
from typing import cast

import streamlit as st

from bizcompass_kh.config.settings import Settings, get_settings
from bizcompass_kh.databases.postgres import Postgres
from bizcompass_kh.services.embedding import create_embedding_service
from bizcompass_kh.services.llm import create_provider
from bizcompass_kh.services.monitoring import (
    RequestLog,
    save_feedback,
    save_request,
    start_metrics_server,
)
from bizcompass_kh.services.rag import Citation, RAGAnswer, RAGService
from bizcompass_kh.services.vector_db import VectorDBService


@dataclass(frozen=True)
class ChatEntry:
    question: str
    answer: RAGAnswer | None = None
    error: str | None = None
    request_id: int | None = None


@st.cache_resource
def build_services() -> tuple[Settings, Postgres, RAGService]:
    settings = get_settings()
    embedder = create_embedding_service(settings)
    embedder.warmup()
    provider = create_provider(settings)
    database = Postgres(settings.postgres_dsn, settings.embedding_dimension)
    database.init_database()
    start_metrics_server(settings.metrics_port)
    retriever = VectorDBService(
        embedder,
        candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.rrf_k,
    )
    rag = RAGService(retriever, provider, retrieval_k=settings.retrieval_k)
    return settings, database, rag


def citation_links(citations: list[Citation]) -> str:
    references: list[str] = []
    for index, citation in enumerate(citations, start=1):
        page = f", page {citation.page}" if citation.page else ""
        tooltip = escape(f"{citation.title} — {citation.agency}{page}", quote=True)
        url = escape(citation.url, quote=True)
        references.append(
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" title="{tooltip}">'
            f"[{index}]</a>"
        )
    return " ".join(references)


def citation_markdown(citation: Citation, index: int) -> str:
    page = f", page {citation.page}" if citation.page else ""
    tooltip = escape(f"{citation.title} - {citation.agency}{page}", quote=True)
    url = escape(citation.url, quote=True)
    return f'[{index}]({url} "{tooltip}")'


def answer_with_citations(answer: str, citations: list[Citation]) -> str:
    """Place source links in the answer text without trusting model HTML."""
    citation_by_index = {index: citation for index, citation in enumerate(citations, start=1)}

    def replace_reference(match: re.Match[str]) -> str:
        citation = citation_by_index.get(int(match.group(1)))
        if citation is None:
            return match.group(0)
        return citation_markdown(citation, int(match.group(1)))

    rendered = re.sub(r"\[S?(\d+)\](?!\()", replace_reference, answer)
    return rendered


def render_answer(answer: RAGAnswer) -> None:
    st.markdown(answer_with_citations(answer.answer, answer.citations))


def render_feedback(database: Postgres, request_id: int) -> None:
    feedback_requests = st.session_state.setdefault("feedback_requests", set())
    if request_id in feedback_requests:
        st.caption("Feedback recorded")
        return

    st.markdown('<p class="feedback-label">Was this answer useful?</p>', unsafe_allow_html=True)
    helpful, unhelpful = st.columns(2, gap="small")
    score = None
    with helpful:
        if st.button(
            "Helpful",
            icon=":material/thumb_up:",
            key=f"helpful-{request_id}",
            use_container_width=True,
        ):
            score = 1
    with unhelpful:
        if st.button(
            "Not helpful",
            icon=":material/thumb_down:",
            key=f"unhelpful-{request_id}",
            use_container_width=True,
        ):
            score = -1

    if score is not None:
        with database.session() as session:
            save_feedback(session, request_id, score)
        feedback_requests.add(request_id)
        st.rerun()


def get_history() -> list[ChatEntry]:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    return cast(list[ChatEntry], st.session_state["chat_history"])


def user_message_html(question: str) -> str:
    escaped_question = escape(question)
    return (
        '<div class="user-message"><div class="user-message-bubble">'
        f"{escaped_question}</div></div>"
    )


def render_user_message(question: str) -> None:
    st.markdown(user_message_html(question), unsafe_allow_html=True)


def render_history(history: list[ChatEntry], database: Postgres) -> None:
    for entry in history:
        render_user_message(entry.question)
        with st.chat_message("assistant", avatar=":material/account_balance:"):
            if entry.answer:
                render_answer(entry.answer)
                if entry.request_id is not None and entry.answer.citations and entry.answer.answer:
                    render_feedback(database, entry.request_id)
            elif entry.error:
                st.error(entry.error)


def main() -> None:
    st.set_page_config(page_title="BizCompass KH", page_icon="🇰🇭")
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] { max-width: 860px; padding-top: 2.5rem; }
        [data-testid="stChatMessage"] {
          border: 0;
          border-bottom: 1px solid rgba(128, 145, 135, 0.16);
          border-radius: 0;
          padding: 0.35rem 0 1rem;
          margin-bottom: 1rem;
        }
        .user-message {
          display: flex;
          justify-content: flex-end;
          margin: 0.35rem 0 1rem;
        }
        .user-message-bubble {
          background: #e1efe4;
          border: 1px solid #c8ddcd;
          border-radius: 16px 4px 16px 16px;
          color: #173522;
          max-width: 72%;
          padding: 0.7rem 0.9rem;
          text-align: left;
        }
        [data-testid="stChatMessage"] a {
          color: #1d8a5b;
          font-weight: 700;
          text-decoration: none;
          border-bottom: 1px solid currentColor;
        }
        [data-testid="stChatMessage"] a:hover, [data-testid="stChatMessage"] a:focus {
          color: #65bd83;
          border-bottom-width: 2px;
        }
        .feedback-label {
          color: #5b6b62;
          font-size: 0.78rem;
          font-weight: 600;
          margin: 0.85rem 0 0.35rem;
        }
        [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] { max-width: 280px; }
        [data-testid="stChatInput"] { margin-top: 1.5rem; }
        [data-testid="stChatInput"] textarea { border-radius: 12px; }
        .app-eyebrow {
          color: #1d8a5b;
          font-size: 0.78rem;
          font-weight: 750;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          margin-bottom: 0.35rem;
        }
        .app-subtitle { color: #8d9a92; margin-top: -0.55rem; margin-bottom: 1.5rem; }
        @media (max-width: 640px) {
          [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("BizCompass KH")
        st.caption("Cambodia SME & business guidance")
        if st.button("New conversation", use_container_width=True):
            st.session_state.pop("chat_history", None)
            st.session_state.pop("last_request_id", None)
            st.rerun()
        st.divider()
        st.caption("This conversation stays in this browser session and clears on refresh.")

    st.markdown('<p class="app-eyebrow">Cambodia business guidance</p>', unsafe_allow_html=True)
    st.title("BizCompass KH")
    st.markdown(
        '<p class="app-subtitle">Practical, evidence-backed SME and registration guidance.</p>',
        unsafe_allow_html=True,
    )

    try:
        settings, database, rag = build_services()
    except Exception as error:
        st.error(f"Startup failed: {error}")
        st.stop()

    history = get_history()
    if history:
        render_history(history, database)
    else:
        with st.container(border=True):
            st.subheader("Start with a business question")
            st.write(
                "Ask about registration, licensing, or operating an SME in Cambodia. "
                "Answers are grounded in the knowledge base."
            )

    question = st.chat_input("Ask about starting or registering a business in Cambodia")
    if question:
        render_user_message(question)
        started = time.perf_counter()
        try:
            with st.chat_message("assistant", avatar=":material/account_balance:"), st.spinner(
                "Finding grounded guidance..."
            ):
                with database.session() as session:
                    answer = rag.answer(session, question)
                    request_id = save_request(
                        session,
                        RequestLog(
                            question,
                            settings.llm_provider,
                            settings.llm_model,
                            (time.perf_counter() - started) * 1000,
                            result=answer,
                        ),
                    )
                render_answer(answer)
                if answer.citations and answer.answer:
                    render_feedback(database, request_id)
            history.append(ChatEntry(question, answer, request_id=request_id))
        except Exception as error:
            with database.session() as session:
                save_request(
                    session,
                    RequestLog(
                        question,
                        settings.llm_provider,
                        settings.llm_model,
                        (time.perf_counter() - started) * 1000,
                        error=str(error),
                    ),
                )
            message = f"Request failed: {error}"
            history.append(ChatEntry(question, error=message))
            with st.chat_message("assistant", avatar=":material/account_balance:"):
                st.error(message)

if __name__ == "__main__":
    main()
