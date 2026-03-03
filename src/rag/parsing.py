import logging
import re
from pathlib import Path

import yaml

from rag.config import MAX_INCLUDE_DEPTH
from rag.models import Document, ImportGraph

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)?---\n?", flags=re.DOTALL)


# --- Text utilities ---


def extract_frontmatter(text: str) -> dict:
    """Extract YAML frontmatter as a dict. Returns {} if none found."""
    match = _FRONTMATTER_RE.match(text)
    if not match or not match.group(1):
        return {}
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        logger.warning("Failed to parse YAML frontmatter")
        return {}


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (--- delimited block at start of file)."""
    return _FRONTMATTER_RE.sub("", text, count=1)


def extract_between_markers(text: str, start: str, end: str) -> str:
    """Extract text between start and end marker lines (exclusive).

    Used to pull code snippets from source files (.py, .ts, etc.) that
    use comment markers like "# START Example" / "# END Example".
    End markers are checked before start markers on each line, so if both
    appear on the same line the section is treated as empty.
    Uses 'in' matching because Weaviate source files can have multiple
    markers on one line (e.g. "# END Foo  # END Bar").
    """
    lines = text.splitlines(keepends=True)
    collecting = False
    result = []
    for line in lines:
        if end in line:
            collecting = False
        elif start in line:
            collecting = True
        elif collecting:
            result.append(line)
    return "".join(result)


def strip_mdx_syntax(text: str) -> str:
    """Remove MDX/JSX syntax, keeping plain markdown content.

    Strips import statements, HTML comments, and JSX/HTML tags.
    Converts Docusaurus admonition markers (:::tip etc.) to bold
    labels so the semantic meaning is preserved for RAG.
    Should be called after resolve_filtered_text_blocks.

    Preserves content inside fenced code blocks (``` or ~~~).
    """
    # Split text into code blocks and non-code segments so we only
    # transform the non-code parts.
    parts = re.split(r"(^(?:`{3,}|~{3,}).*$\n(?:.*\n)*?^(?:`{3,}|~{3,})\s*$)", text, flags=re.MULTILINE)

    for i, part in enumerate(parts):
        # Odd-indexed parts are the captured code blocks — skip them
        if i % 2 == 1:
            continue
        # Remove JS/MDX import statements (import X from '...' or import { X } from '...')
        part = re.sub(
            r"^import\s+(?:\w+|\{[^}]+\})\s+from\s+['\"].*['\"]\s*;?\s*$\n?",
            "",
            part,
            flags=re.MULTILINE,
        )
        # Remove HTML comments (possibly multiline)
        part = re.sub(r"<!--.*?-->", "", part, flags=re.DOTALL)
        # Convert admonition openers to bold labels
        # :::tip → **Tip:**  |  :::info Glossary → **Glossary:**
        part = re.sub(
            r"^:::(tip|note|warning|caution|danger|info)(?:[ \t]+(.+))?$",
            lambda m: f"**{m.group(2) or m.group(1).title()}:**",
            part,
            flags=re.MULTILINE,
        )
        # Remove admonition closing markers
        part = re.sub(r"^:::$\n?", "", part, flags=re.MULTILINE)
        # Convert <TabItem> labels to bold headings before stripping JSX
        part = re.sub(
            r'<TabItem\s[^>]*label="([^"]+)"[^>]*>',
            lambda m: f"**{m.group(1)}:**\n",
            part,
        )
        # Remove JSX/HTML tags but keep text content between them
        part = re.sub(r"</?[A-Za-z][^>]*>", "", part)
        parts[i] = part

    text = "".join(parts)
    # Clean up excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# --- Import parsers ---


def resolve_import_path(import_path: str, file_path: Path, repo_dir: Path) -> Path:
    """Resolve an import path string to an absolute Path.

    Absolute paths (/) resolve from repo_dir, relative (./ ../) from file_path,
    bare paths resolve from repo_dir.
    """
    import_path = import_path.replace("\\_", "_")
    if import_path.startswith("/"):
        return (repo_dir / import_path.lstrip("/")).resolve()
    elif import_path.startswith("./") or import_path.startswith("../"):
        return (file_path.parent / import_path).resolve()
    return (repo_dir / import_path).resolve()


def parse_file_imports(text: str, file_path: Path, repo_dir: Path) -> list[Path]:
    """Parse all file imports (MDX includes and raw-loader) and resolve to paths.

    Used by build_import_graph to determine the dependency tree between files.
    Returns resolved Paths but discards variable names.
    See also: parse_raw_imports, which preserves variable names for
    resolving FilteredTextBlock components.
    """
    imports = []
    for match in re.finditer(
        r"^import\s+\w+\s+from\s+['\"](?:!!raw-loader!)?(.+?\.(?:mdx?|py|ts|js|go|java))['\"]",
        text,
        flags=re.MULTILINE,
    ):
        resolved = resolve_import_path(match.group(1), file_path, repo_dir)
        if resolved.exists():
            imports.append(resolved)
    return imports


def parse_raw_imports(text: str) -> dict[str, str]:
    """Parse !!raw-loader! import statements, returning {variable_name: file_path}.

    Used to resolve FilteredTextBlock components, which reference code
    by variable name (e.g. text={EndToEndPyCode}).
    See also: parse_file_imports, which resolves all imports to Paths
    for building the dependency graph.
    """
    imports = {}
    for match in re.finditer(
        r"^import\s+(\w+)\s+from\s+['\"]!!raw-loader!(.+?)['\"]",
        text,
        flags=re.MULTILINE,
    ):
        imports[match.group(1)] = match.group(2)
    return imports


def parse_mdx_imports(text: str, file_path: Path, repo_dir: Path) -> dict[str, Path]:
    """Parse MDX file imports, returning {component_name: resolved_path}.

    Only includes imports that point to actual .md/.mdx files on disk.
    Excludes !!raw-loader! imports and @theme imports.
    Used by inline_mdx_includes to replace <ComponentName /> with file content.
    """
    imports = {}
    for match in re.finditer(
        r"^import\s+(\w+)\s+from\s+['\"]([^'\"!@]+\.mdx?)['\"]\s*;?\s*$",
        text,
        flags=re.MULTILINE,
    ):
        resolved = resolve_import_path(match.group(2), file_path, repo_dir)
        if resolved.exists():
            imports[match.group(1)] = resolved
    return imports


# --- Document parsing ---


def parse_document_text(text: str, file_path: Path) -> Document:
    """Parse markdown/mdx text into a Document."""
    metadata = extract_frontmatter(text)
    body = strip_frontmatter(text)
    title = metadata.pop("title", file_path.stem)
    return Document(
        title=title,
        body=body,
        source_path=file_path,
        metadata=metadata,
    )


def parse_document(file_path: Path) -> Document | None:
    """Read a markdown/mdx file and parse into a Document. Returns None on read failure."""
    logger.info(f"Parsing {file_path}")
    try:
        text = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        logger.warning(f"Skipping {file_path}: {e}")
        return None
    return parse_document_text(text, file_path)


# --- Content transformation ---


def resolve_filtered_text_blocks(text: str, file_path: Path, repo_dir: Path) -> str:
    """Replace <FilteredTextBlock> tags with markdown code blocks.

    Parses raw-loader imports to map variable names to source files,
    then finds each FilteredTextBlock tag, reads the source file,
    extracts the code between markers, and replaces the tag with a
    fenced code block.
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    raw_imports = parse_raw_imports(text)
    if not raw_imports:
        return text

    def replace_block(match: re.Match) -> str:
        props = match.group(1)
        # Extract props: text={VarName}, startMarker="...", endMarker="...", language="..."
        var_match = re.search(r"text=\{(\w+)\}", props)
        start_match = re.search(r'startMarker="(.+?)"', props)
        end_match = re.search(r'endMarker="(.+?)"', props)
        lang_match = re.search(r'language="(.+?)"', props)

        if not var_match:
            logger.warning(f"Malformed FilteredTextBlock: {match.group(0)[:80]}")
            return ""

        var_name = var_match.group(1)
        if var_name not in raw_imports:
            logger.warning(f"Unknown import variable: {var_name}")
            return ""

        source_path = resolve_import_path(raw_imports[var_name], file_path, repo_dir)

        try:
            source_text = source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Cannot read source file {source_path}: {e}")
            return ""

        if start_match and end_match:
            code = extract_between_markers(
                source_text, start_match.group(1), end_match.group(1)
            )
        else:
            code = source_text
        lang = lang_match.group(1) if lang_match else ""
        return f"```{lang}\n{code}```\n"

    # Match <FilteredTextBlock ... /> (self-closing, possibly multiline)
    return re.sub(
        r"<FilteredTextBlock\s+(.*?)\s*/>",
        replace_block,
        text,
        flags=re.DOTALL,
    )


def inline_mdx_includes(text: str, file_path: Path, repo_dir: Path, _depth: int = 0) -> str:
    """Replace MDX component usages with the content of the imported files.

    For each MDX import like `import Intro from './_intro.mdx'`, finds
    usages of `<Intro />` and replaces them with the file's content.
    Recurses into included files to handle nested includes (max depth 3).
    """
    if _depth > MAX_INCLUDE_DEPTH:
        logger.error(f"Max include depth exceeded at {file_path}")
        return text

    mdx_imports = parse_mdx_imports(text, file_path, repo_dir)
    if not mdx_imports:
        return text

    for name, path in mdx_imports.items():
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Cannot read include file {path}: {e}")
            content = ""

        # Strip frontmatter from included file
        content = strip_frontmatter(content)
        # Recursively inline includes in the included content
        content = inline_mdx_includes(content, path, repo_dir, _depth + 1)
        # Replace self-closing usage: <Name />
        text = re.sub(rf"<{name}\s*/>", content, text)

    return text


# --- Orchestrators ---


def build_import_graph(repo_dir: Path) -> ImportGraph:
    """Scan all md/mdx files and build an import dependency graph."""
    graph = ImportGraph()
    repo_dir = repo_dir.resolve()
    all_files = sorted(f for f in repo_dir.rglob("*") if f.suffix in (".md", ".mdx"))
    logger.info(f"Building import graph from {len(all_files)} files")
    for f in all_files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            logger.warning(f"Skipping {f}: {e}")
            continue
        imports = parse_file_imports(text, f, repo_dir)
        graph.pages[f] = imports
    logger.info(
        f"Import graph built: {len(graph.pages)} pages, {len(graph.top_level_pages())} top-level"
    )
    return graph


def resolve_document(page_path: Path, repo_dir: Path) -> Document | None:
    """Fully resolve a top-level page into a clean Document.

    Ties together the full pipeline:
    1. Parse the file (extract frontmatter, strip it)
    2. Inline content from imported MDX includes
    3. Resolve FilteredTextBlocks into code blocks
    4. Strip remaining MDX syntax
    """
    doc = parse_document(page_path)
    if doc is None:
        return None

    text = doc.body
    text = inline_mdx_includes(text, page_path, repo_dir)
    text = resolve_filtered_text_blocks(text, page_path, repo_dir)
    text = strip_mdx_syntax(text)

    return Document(
        title=doc.title,
        body=text,
        source_path=page_path,
        metadata=doc.metadata,
    )
