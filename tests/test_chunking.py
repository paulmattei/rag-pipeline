from rag.chunking import (
    build_breadcrumbs,
    create_chunks,
    format_chunk_text,
    header_level,
    split_at_paragraphs,
    split_by_headers,
)


def test_500_char_splits_evenly():
    text = "a" * 1000
    chunks = create_chunks(text, "title", "path", strategy="500-char")
    assert len(chunks) == 2
    assert all(len(c.text) == 500 for c in chunks)


def test_500_char_runt_chunk():
    text = "a" * 700
    chunks = create_chunks(text, "title", "path", strategy="500-char")
    assert len(chunks) == 2
    assert len(chunks[0].text) == 500
    assert len(chunks[1].text) == 200


def test_500_char_with_runt_check_merges_runt():
    text = "a" * 700
    chunks = create_chunks(text, "title", "path", strategy="500-char-with-runt-check")
    assert len(chunks) == 1
    assert len(chunks[0].text) == 700


def test_500_char_with_runt_check_no_merge_when_exact():
    text = "a" * 1000
    chunks = create_chunks(text, "title", "path", strategy="500-char-with-runt-check")
    assert len(chunks) == 2
    assert all(len(c.text) == 500 for c in chunks)


def test_document_strategy_single_chunk():
    text = "a" * 2000
    chunks = create_chunks(text, "title", "path", strategy="document")
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_metadata():
    text = "a" * 1200
    chunks = create_chunks(text, "My Doc", "/docs/page.md", strategy="500-char")
    assert len(chunks) == 3
    for i, chunk in enumerate(chunks):
        assert chunk.document_title == "My Doc"
        assert chunk.source_path == "/docs/page.md"
        assert chunk.chunk_index == i


def test_invalid_strategy_raises():
    try:
        create_chunks("text", "title", "path", strategy="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_empty_text():
    chunks = create_chunks("", "title", "path", strategy="500-char")
    assert len(chunks) == 0


def test_short_text_no_split():
    text = "hello"
    chunks = create_chunks(text, "title", "path", strategy="500-char")
    assert len(chunks) == 1
    assert chunks[0].text == "hello"


# --- header_level ---


def test_header_level():
    assert header_level("# Title") == 1
    assert header_level("## Section") == 2
    assert header_level("### Sub") == 3
    assert header_level("#### Deep") == 4
    assert header_level("Not a header") == 0
    assert header_level("#NoSpace") == 0
    assert header_level("") == 0


# --- split_by_headers ---


def test_split_by_headers_basic():
    text = "intro\n\n## Section A\nbody a\n\n### Sub A1\nbody a1\n\n## Section B\nbody b"
    sections = split_by_headers(text)
    assert len(sections) == 4
    assert sections[0] == ("", "intro")
    assert sections[1] == ("## Section A", "body a")
    assert sections[2] == ("### Sub A1", "body a1")
    assert sections[3] == ("## Section B", "body b")


def test_split_by_headers_starts_with_header():
    text = "## Only\nBody here."
    sections = split_by_headers(text)
    assert sections == [("## Only", "Body here.")]


def test_split_by_headers_no_headers():
    text = "Just plain text.\nAnother line."
    sections = split_by_headers(text)
    assert sections == [("", "Just plain text.\nAnother line.")]


# --- build_breadcrumbs ---


def test_build_breadcrumbs_hierarchy():
    sections = [
        ("", "intro"),
        ("# Top", "body"),
        ("## Mid", "body"),
        ("### Deep", "body"),
        ("## Other", "body"),
    ]
    breadcrumbs = build_breadcrumbs(sections)
    assert breadcrumbs == ["", "Top", "Top > Mid", "Top > Mid > Deep", "Top > Other"]


def test_build_breadcrumbs_flat():
    sections = [("## A", ""), ("## B", ""), ("## C", "")]
    breadcrumbs = build_breadcrumbs(sections)
    assert breadcrumbs == ["A", "B", "C"]


# --- split_at_paragraphs ---


def test_split_at_paragraphs_basic():
    text = "para one\n\npara two\n\npara three"
    parts = split_at_paragraphs(text, 20)
    assert parts == ["para one\n\npara two", "para three"]


def test_split_at_paragraphs_single_large():
    text = "a" * 100
    parts = split_at_paragraphs(text, 50)
    assert parts == [text]


def test_split_at_paragraphs_all_fit():
    text = "short\n\nalso short"
    parts = split_at_paragraphs(text, 1000)
    assert parts == [text]


# --- format_chunk_text ---


def test_format_chunk_text_with_breadcrumb():
    result = format_chunk_text("Doc", "A > B", "body text")
    assert result == "Doc > A > B\n\nbody text"


def test_format_chunk_text_without_breadcrumb():
    result = format_chunk_text("Doc", "", "body text")
    assert result == "Doc\n\nbody text"


# --- markdown-sections strategy (end-to-end) ---


def test_markdown_sections_basic():
    text = "## Setup\nInstall docker.\n\n## Usage\nRun the command."
    chunks = create_chunks(text, "Guide", "/guide.md", strategy="markdown-sections")
    assert len(chunks) == 2
    assert "Guide > Setup" in chunks[0].text
    assert "Install docker." in chunks[0].text
    assert "Guide > Usage" in chunks[1].text
    assert "Run the command." in chunks[1].text


def test_markdown_sections_empty():
    chunks = create_chunks("", "title", "path", strategy="markdown-sections")
    assert len(chunks) == 0


def test_markdown_sections_no_headers():
    text = "Just some text without any headers."
    chunks = create_chunks(text, "Doc", "/doc.md", strategy="markdown-sections")
    assert len(chunks) == 1
    assert chunks[0].text == "Doc\n\nJust some text without any headers."


def test_markdown_sections_preserves_hierarchy():
    text = "# Top\nIntro\n\n## Mid\nMiddle text\n\n### Deep\nDeep text"
    chunks = create_chunks(text, "Doc", "/doc.md", strategy="markdown-sections")
    contexts = [c.text.split("\n\n")[0] for c in chunks]
    assert "Doc > Top" in contexts
    assert "Doc > Top > Mid" in contexts
    assert "Doc > Top > Mid > Deep" in contexts


def test_markdown_sections_long_section_splits():
    paragraph = "This is a paragraph with enough words to take up meaningful space. " * 5
    body = "\n\n".join(paragraph for _ in range(20))
    text = f"## Big Section\n{body}"
    chunks = create_chunks(text, "Doc", "/doc.md", strategy="markdown-sections")
    assert len(chunks) > 1
    assert all("Doc > Big Section" in c.text for c in chunks)


def test_markdown_sections_skips_empty_body():
    text = "## Header Only\n\n## Also Empty\n\n## Has Body\nActual content."
    chunks = create_chunks(text, "Doc", "/doc.md", strategy="markdown-sections")
    assert len(chunks) == 1
    assert "Has Body" in chunks[0].text


def test_markdown_sections_metadata():
    text = "## A\nBody a\n\n## B\nBody b"
    chunks = create_chunks(text, "My Doc", "/docs/page.md", strategy="markdown-sections")
    for i, chunk in enumerate(chunks):
        assert chunk.document_title == "My Doc"
        assert chunk.source_path == "/docs/page.md"
        assert chunk.chunk_index == i
