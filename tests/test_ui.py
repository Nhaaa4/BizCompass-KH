from bizcompass_kh.services.rag import Citation
from bizcompass_kh.ui import app
from bizcompass_kh.ui.app import citation_links


def test_citation_links_are_compact_clickable_references() -> None:
    links = citation_links(
        [
            Citation(
                title="Business Registration Guide",
                agency="Ministry of Commerce",
                url="https://example.com/guide.pdf",
                page=4,
            )
        ]
    )

    assert 'href="https://example.com/guide.pdf"' in links
    assert 'title="Business Registration Guide — Ministry of Commerce, page 4"' in links
    assert ">[1]</a>" in links


def test_answer_citations_replace_matching_inline_reference() -> None:
    citation = Citation(
        title="Business Registration Guide",
        agency="Ministry of Commerce",
        url="https://example.com/guide.pdf",
        page=4,
    )

    rendered = app.answer_with_citations("Register the business first [1].", [citation])

    assert "Register the business first" in rendered
    citation_link = (
        '[1](https://example.com/guide.pdf '
        '"Business Registration Guide - Ministry of Commerce, page 4")'
    )
    assert citation_link in rendered
    assert "Sources:" not in rendered


def test_answer_citations_replaces_legacy_evidence_label() -> None:
    citation = Citation(
        title="Business Registration Guide",
        agency="Ministry of Commerce",
        url="https://example.com/guide.pdf",
        page=4,
    )

    rendered = app.answer_with_citations("Register online [S1].", [citation])

    citation_link = (
        '[1](https://example.com/guide.pdf '
        '"Business Registration Guide - Ministry of Commerce, page 4")'
    )
    assert citation_link in rendered


def test_answer_citations_preserves_markdown_link_written_by_ai() -> None:
    answer = "Register online [1](https://example.com/guide.pdf)."
    citation = Citation(
        title="Business Registration Guide",
        agency="Ministry of Commerce",
        url="https://example.com/guide.pdf",
        page=4,
    )

    rendered = app.answer_with_citations(answer, [citation])

    assert rendered == answer


def test_answer_citations_does_not_add_sources_when_model_omits_references() -> None:
    citation = Citation(
        title="Business Registration Guide",
        agency="Ministry of Commerce",
        url="https://example.com/guide.pdf",
        page=4,
    )

    rendered = app.answer_with_citations("Register the business first.", [citation])

    assert rendered == "Register the business first."


def test_user_message_html_escapes_content_and_marks_the_message_as_user() -> None:
    rendered = app.user_message_html("How do I register <a business>?")

    assert 'class="user-message"' in rendered
    assert "How do I register &lt;a business&gt;?" in rendered
