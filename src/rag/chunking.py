import re

from rag.models import Chunk


def create_chunks(text, document_title, source_path, strategy="500-char"):
    if strategy == "500-char":
        return create_chunks_500_char(text, document_title, source_path)
    elif strategy == "500-char-with-runt-check":
        return create_chunks_500_char_with_runt_check(text, document_title, source_path)
    elif strategy == "document":
        return create_chunks_document(text, document_title, source_path)
    elif strategy == "markdown-sections":
        return create_chunks_markdown_sections(text, document_title, source_path)
    elif strategy == "markdown-sections-no-breadcrumbs":
        return create_chunks_markdown_sections(text, document_title, source_path, use_breadcrumbs=False)
    elif strategy == "markdown-optimized":
        return create_chunks_markdown_optimized(text, document_title, source_path)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")


# Fixed size 500 char chunks, no overlap. Append runt chunk to penultimate chunk if it's less than 500 chars.
# We also maintain the document title and path for each chunk.
def create_chunks_500_char(text, document_title, source_path):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    chunks = [Chunk(text=chunk, document_title=document_title, source_path=source_path, chunk_index=i) for i, chunk in enumerate(chunks)]
    return chunks


# Fixed size 500 char chunks, no overlap. Append runt chunk to penultimate chunk if it's less than 500 chars.
def create_chunks_500_char_with_runt_check(text, document_title, source_path):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    if len(chunks) >= 2 and len(chunks[-1]) < 500:
        chunks[-2] += chunks[-1]
        chunks.pop()
    chunks = [Chunk(text=chunk, document_title=document_title, source_path=source_path, chunk_index=i) for i, chunk in enumerate(chunks)]
    return chunks


# A chunk is the entire document
def create_chunks_document(text, document_title, source_path):
    return [Chunk(text=text, document_title=document_title, source_path=source_path, chunk_index=0)]


def create_chunks_markdown_sections(text, document_title, source_path, max_section_chars=2000, use_breadcrumbs=True):
    """Chunk by markdown sections with header breadcrumb context.

    Splits at header boundaries, prefixes each chunk with its document title
    and header hierarchy, and splits oversized sections at paragraph boundaries.
    """
    if not text.strip():
        return []

    sections = split_by_headers(text)
    breadcrumbs = build_breadcrumbs(sections) if use_breadcrumbs else [""] * len(sections)

    texts = []
    for (_header, body), breadcrumb in zip(sections, breadcrumbs):
        if not body:
            continue
        if len(body) > max_section_chars:
            for part in split_at_paragraphs(body, max_section_chars):
                texts.append(format_chunk_text(document_title, breadcrumb, part))
        else:
            texts.append(format_chunk_text(document_title, breadcrumb, body))

    return [
        Chunk(text=t, document_title=document_title, source_path=source_path, chunk_index=i)
        for i, t in enumerate(texts)
    ]


# --- helpers ---


def header_level(line):
    """Return the markdown header level (1-6), or 0 if not a header."""
    match = re.match(r"^(#{1,6})\s", line)
    return len(match.group(1)) if match else 0


def split_by_headers(text):
    """Split markdown text into sections at header boundaries.

    Returns list of (header_line, body_text) tuples. The first section
    has an empty header if the text starts with body content.
    """
    sections = []
    current_header = ""
    body_lines = []

    for line in text.split("\n"):
        if header_level(line) > 0:
            body = "\n".join(body_lines).strip()
            if current_header or body:
                sections.append((current_header, body))
            current_header = line
            body_lines = []
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()
    if current_header or body:
        sections.append((current_header, body))

    return sections


def build_breadcrumbs(sections):
    """Build ancestor header trail for each section.

    Returns list of breadcrumb strings like "Installation > Docker > Setup".
    """
    breadcrumbs = []
    stack = []

    for header, _body in sections:
        level = header_level(header)
        if level > 0:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, header.lstrip("#").strip()))
            breadcrumbs.append(" > ".join(text for _, text in stack))
        else:
            breadcrumbs.append("")

    return breadcrumbs


def split_at_paragraphs(text, max_chars):
    """Split text at paragraph boundaries, keeping parts under max_chars.

    A single paragraph longer than max_chars is kept intact.
    """
    paragraphs = text.split("\n\n")
    parts = []
    current = []
    current_length = 0

    for paragraph in paragraphs:
        added_length = len(paragraph) + (2 if current else 0)
        if current and current_length + added_length > max_chars:
            parts.append("\n\n".join(current))
            current = [paragraph]
            current_length = len(paragraph)
        else:
            current.append(paragraph)
            current_length += added_length

    if current:
        parts.append("\n\n".join(current))

    return parts


def source_category(source_path):
    """Extract a category from the docs directory structure.

    e.g. 'deploy > installation-guides' from '.../docs/deploy/installation-guides/embedded.md'
    """
    path_str = str(source_path)
    if "/docs/" in path_str:
        relative = path_str.split("/docs/", 1)[1]
        parts = relative.split("/")
        directories = parts[:-1][:2]
        return " > ".join(directories) if directories else ""
    return ""


def format_chunk_text(document_title, breadcrumb, body, category=""):
    """Combine document title, breadcrumb, and body into embeddable chunk text."""
    context = f"{document_title} > {breadcrumb}" if breadcrumb else document_title
    if category:
        context = f"[{category}] {context}"
    return f"{context}\n\n{body}"


def _hard_split_text(text, max_chars):
    """Split text at word boundaries to stay under max_chars."""
    pieces = []
    while len(text) > max_chars:
        split_pos = text.rfind(" ", 0, max_chars)
        if split_pos <= 0:
            split_pos = max_chars
        pieces.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    if text:
        pieces.append(text)
    return pieces


def create_chunks_markdown_optimized(text, document_title, source_path,
                                     max_child_chars=1500, min_child_chars=200):
    """Markdown-sections with size cap, runt merging, and parent-child retrieval.

    Child chunks are sized for embedding models (~1500 chars ≈ 400 tokens).
    Each child stores its parent section text for richer LLM context at query time.
    """
    if not text.strip():
        return []

    sections = split_by_headers(text)
    breadcrumbs = build_breadcrumbs(sections)

    # Build parent sections and split into children
    children = []
    for (header, body), breadcrumb in zip(sections, breadcrumbs):
        if not body:
            continue
        parent_text = format_chunk_text(document_title, breadcrumb, body)

        # Fix 1: account for breadcrumb prefix in size budget
        prefix_length = len(format_chunk_text(document_title, breadcrumb, ""))
        body_budget = max(200, max_child_chars - prefix_length)

        if len(body) <= body_budget:
            child_text = format_chunk_text(document_title, breadcrumb, body)
            children.append((child_text, parent_text))
        else:
            for part in split_at_paragraphs(body, body_budget):
                # Fix 2: hard-split oversized single paragraphs
                if len(part) > body_budget:
                    for sub in _hard_split_text(part, body_budget):
                        child_text = format_chunk_text(document_title, breadcrumb, sub)
                        children.append((child_text, parent_text))
                else:
                    child_text = format_chunk_text(document_title, breadcrumb, part)
                    children.append((child_text, parent_text))

    # Merge small adjacent children
    merged = []
    i = 0
    while i < len(children):
        child_text, parent_text = children[i]
        while i + 1 < len(children) and len(child_text) < min_child_chars:
            next_text, next_parent = children[i + 1]
            child_text = child_text + "\n\n" + next_text
            if next_parent != parent_text:
                parent_text = parent_text + "\n\n---\n\n" + next_parent
            i += 1
        merged.append((child_text, parent_text))
        i += 1

    # Fix 3: merge trailing runt backward
    if len(merged) >= 2 and len(merged[-1][0]) < min_child_chars:
        last_text, last_parent = merged.pop()
        previous_text, previous_parent = merged[-1]
        merged_parent = previous_parent if last_parent == previous_parent else previous_parent + "\n\n---\n\n" + last_parent
        merged[-1] = (previous_text + "\n\n" + last_text, merged_parent)

    return [
        Chunk(text=child_text, document_title=document_title,
              source_path=source_path, chunk_index=i, parent_text=parent_text)
        for i, (child_text, parent_text) in enumerate(merged)
    ]
