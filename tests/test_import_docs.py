from pathlib import Path

from rag.parsing import (
    build_import_graph,
    extract_between_markers,
    extract_frontmatter,
    inline_mdx_includes,
    parse_file_imports,
    parse_mdx_imports,
    parse_raw_imports,
    resolve_document,
    resolve_filtered_text_blocks,
    strip_frontmatter,
    strip_mdx_syntax,
)


def test_extract_frontmatter():
    text = "---\ntitle: Hello\nsidebar_position: 3\n---\n# Content\n"
    assert extract_frontmatter(text) == {
        "title": "Hello",
        "sidebar_position": 3,
    }


def test_extract_frontmatter_with_lists():
    text = "---\ntitle: My Page\ntags: [search, api]\n---\n# Content\n"
    meta = extract_frontmatter(text)
    assert meta["title"] == "My Page"
    assert meta["tags"] == ["search", "api"]


def test_extract_frontmatter_none():
    assert extract_frontmatter("# No frontmatter") == {}


def test_extract_frontmatter_empty():
    assert extract_frontmatter("---\n---\n# Content") == {}


def test_strips_frontmatter():
    text = "---\ntitle: Hello\nsidebar_position: 3\n---\n# Content here\n"
    assert strip_frontmatter(text) == "# Content here\n"


def test_no_frontmatter():
    text = "# Just markdown\nNo frontmatter here.\n"
    assert strip_frontmatter(text) == text


def test_empty_frontmatter():
    text = "---\n---\n# Content\n"
    assert strip_frontmatter(text) == "# Content\n"


def test_dashes_in_body_not_stripped():
    text = "---\ntitle: Hello\n---\n# Content\n---\nThis is a divider\n"
    assert strip_frontmatter(text) == "# Content\n---\nThis is a divider\n"


def test_empty_string():
    assert strip_frontmatter("") == ""


def test_extract_between_markers():
    text = (
        "# START Example\n"
        "import weaviate\n"
        "client = weaviate.Client()\n"
        "# END Example\n"
        "other stuff\n"
    )
    result = extract_between_markers(text, "# START Example", "# END Example")
    assert result == "import weaviate\nclient = weaviate.Client()\n"


def test_extract_between_markers_comment_style():
    text = (
        "// InstantiationExample\n"
        "const client = weaviate.client()\n"
        "// END InstantiationExample\n"
    )
    result = extract_between_markers(
        text, "// InstantiationExample", "// END InstantiationExample"
    )
    assert result == "const client = weaviate.client()\n"


def test_extract_between_markers_not_found():
    text = "just some code\nno markers here\n"
    assert extract_between_markers(text, "# START", "# END") == ""


def test_extract_between_markers_multiple_sections():
    text = "# Example  # Other\nshared line\n# END Example  # END Other\n"
    result = extract_between_markers(text, "# Example", "# END Example")
    assert result == "shared line\n"


def test_extract_between_markers_start_and_end_same_line():
    text = "# Example  # END Example\nthis should not be collected\n"
    assert extract_between_markers(text, "# Example", "# END Example") == ""


def test_extract_between_markers_only_start():
    text = "# START Example\ncode here\nmore code\n"
    assert (
        extract_between_markers(text, "# START Example", "# END Example")
        == "code here\nmore code\n"
    )


def test_parse_file_imports_absolute(tmp_path):
    # Create a fake repo with an include file
    include = tmp_path / "_includes" / "code.mdx"
    include.parent.mkdir(parents=True)
    include.write_text("# code")
    page = tmp_path / "docs" / "page.mdx"
    page.parent.mkdir(parents=True)
    page.write_text("import Code from '/_includes/code.mdx';\n")
    result = parse_file_imports(page.read_text(), page, tmp_path)
    assert result == [include.resolve()]


def test_parse_file_imports_relative(tmp_path):
    sibling = tmp_path / "docs" / "_local.mdx"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("# local")
    page = tmp_path / "docs" / "page.mdx"
    page.write_text("import Local from './_local.mdx';\n")
    result = parse_file_imports(page.read_text(), page, tmp_path)
    assert result == [sibling.resolve()]


def test_parse_file_imports_skips_theme_imports(tmp_path):
    page = tmp_path / "page.mdx"
    page.write_text("import Tabs from '@theme/Tabs';\n")
    result = parse_file_imports(page.read_text(), page, tmp_path)
    assert result == []


def test_import_graph_top_level(tmp_path):
    include = tmp_path / "_includes" / "footer.mdx"
    include.parent.mkdir(parents=True)
    include.write_text("# footer")
    page = tmp_path / "page.mdx"
    page.write_text("import Footer from '/_includes/footer.mdx';\n")
    graph = build_import_graph(tmp_path)
    top = graph.top_level_pages()
    assert page.resolve() in top
    assert include.resolve() not in top


def test_import_graph_no_imports(tmp_path):
    page1 = tmp_path / "a.md"
    page1.write_text("# Page A")
    page2 = tmp_path / "b.md"
    page2.write_text("# Page B")
    graph = build_import_graph(tmp_path)
    top = graph.top_level_pages()
    assert len(top) == 2


def test_resolve_filtered_text_blocks(tmp_path):
    # Create a source file with markers
    code_dir = tmp_path / "_includes" / "code"
    code_dir.mkdir(parents=True)
    source = code_dir / "example.py"
    source.write_text(
        "# Setup\nimport os\n# END Setup\n"
        "# Example\nimport weaviate\nclient = weaviate.Client()\n# END Example\n"
    )
    mdx_text = (
        "import PyCode from '!!raw-loader!/_includes/code/example.py';\n"
        "\n"
        "# My Page\n"
        "\n"
        "<FilteredTextBlock\n"
        "  text={PyCode}\n"
        '  startMarker="# Example"\n'
        '  endMarker="# END Example"\n'
        '  language="py"\n'
        "/>\n"
    )
    result = resolve_filtered_text_blocks(mdx_text, tmp_path / "page.md", tmp_path)
    assert "```py\nimport weaviate\nclient = weaviate.Client()\n```" in result
    assert "FilteredTextBlock" not in result
    assert "# My Page" in result


def test_resolve_filtered_text_blocks_no_imports():
    text = "# Just markdown\nNo imports here.\n"
    assert resolve_filtered_text_blocks(text, Path("page.md"), Path(".")) == text


def test_resolve_filtered_text_blocks_missing_source(tmp_path):
    mdx_text = (
        "import PyCode from '!!raw-loader!/_includes/code/missing.py';\n"
        '<FilteredTextBlock text={PyCode} startMarker="# S" endMarker="# E" language="py" />\n'
    )
    result = resolve_filtered_text_blocks(mdx_text, tmp_path / "page.md", tmp_path)
    assert "FilteredTextBlock" not in result


def test_parse_single_raw_import():
    text = "import EndToEndPyCode from '!!raw-loader!/_includes/code/quickstart/endtoend.py';\n"
    assert parse_raw_imports(text) == {
        "EndToEndPyCode": "/_includes/code/quickstart/endtoend.py",
    }


def test_parse_multiple_raw_imports():
    text = (
        "import Tabs from '@theme/Tabs';\n"
        "import FilteredTextBlock from '@site/src/components/Documentation/FilteredTextBlock';\n"
        "import EndToEndPyCode from '!!raw-loader!/_includes/code/quickstart/endtoend.py';\n"
        "import GoCode from '!!raw-loader!/_includes/code/quickstart/go-connect.go';\n"
    )
    result = parse_raw_imports(text)
    assert result == {
        "EndToEndPyCode": "/_includes/code/quickstart/endtoend.py",
        "GoCode": "/_includes/code/quickstart/go-connect.go",
    }


def test_parse_no_raw_imports():
    text = "import Tabs from '@theme/Tabs';\nimport TabItem from '@theme/TabItem';\n"
    assert parse_raw_imports(text) == {}


def test_parse_double_quote_imports():
    text = 'import MyCode from "!!raw-loader!/_includes/code/example.py";\n'
    assert parse_raw_imports(text) == {
        "MyCode": "/_includes/code/example.py",
    }


def test_strip_mdx_imports():
    text = "import Tabs from '@theme/Tabs';\nimport TabItem from '@theme/TabItem';\n\n# Content\n"
    result = strip_mdx_syntax(text)
    assert "import" not in result
    assert "# Content" in result


def test_strip_mdx_html_comments():
    text = "# Title\n<!-- truncate -->\nSome content\n"
    result = strip_mdx_syntax(text)
    assert "<!--" not in result
    assert "# Title" in result
    assert "Some content" in result


def test_strip_mdx_admonition_to_bold():
    text = ":::tip\nThis is a tip.\n:::\n"
    result = strip_mdx_syntax(text)
    assert "**Tip:**" in result
    assert "This is a tip." in result
    assert ":::" not in result


def test_strip_mdx_admonition_with_custom_title():
    text = ":::info Glossary\n- Node: a machine\n:::\n"
    result = strip_mdx_syntax(text)
    assert "**Glossary:**" in result
    assert "- Node: a machine" in result


def test_strip_mdx_jsx_tags():
    text = '<Tabs groupId="languages">\n<TabItem value="py" label="Python">\ncode here\n</TabItem>\n</Tabs>\n'
    result = strip_mdx_syntax(text)
    assert "code here" in result
    assert "<Tabs" not in result
    assert "<TabItem" not in result
    assert "**Python:**" in result


def test_strip_mdx_tabitem_preserves_language_label():
    text = (
        '<Tabs groupId="languages">\n'
        '<TabItem value="py" label="Python Client v4">\n\n'
        "```python\nclient.connect()\n```\n\n"
        "</TabItem>\n"
        '<TabItem value="ts" label="TypeScript">\n\n'
        "```ts\nclient.connect();\n```\n\n"
        "</TabItem>\n"
        "</Tabs>\n"
    )
    result = strip_mdx_syntax(text)
    assert "**Python Client v4:**" in result
    assert "**TypeScript:**" in result
    assert "```python" in result
    assert "```ts" in result


def test_strip_mdx_self_closing_tags():
    text = "# Page\n<WhatNext />\n"
    result = strip_mdx_syntax(text)
    assert "# Page" in result
    assert "WhatNext" not in result


def test_strip_mdx_preserves_markdown():
    text = "# Title\n\nSome **bold** and *italic* text.\n\n- List item\n"
    assert strip_mdx_syntax(text) == text


def test_strip_mdx_preserves_imports_inside_code_blocks():
    text = (
        "Some text\n"
        "```typescript\n"
        "import { WeaviateClient } from 'weaviate-client';\n"
        "import weaviate from 'weaviate-client';\n"
        "```\n"
        "More text\n"
    )
    assert strip_mdx_syntax(text) == text


def test_parse_mdx_imports(tmp_path):
    include = tmp_path / "_includes" / "intro.mdx"
    include.parent.mkdir(parents=True)
    include.write_text("# Intro content")
    page = tmp_path / "docs" / "page.mdx"
    page.parent.mkdir(parents=True)
    text = "import Intro from '/_includes/intro.mdx';\nimport Tabs from '@theme/Tabs';\nimport PyCode from '!!raw-loader!/code.py';\n"
    result = parse_mdx_imports(text, page, tmp_path)
    assert "Intro" in result
    assert result["Intro"] == include.resolve()
    assert "Tabs" not in result
    assert "PyCode" not in result


def test_inline_mdx_includes(tmp_path):
    include = tmp_path / "_includes" / "intro.mdx"
    include.parent.mkdir(parents=True)
    include.write_text("---\ntitle: Intro\n---\nThis is the intro.\n")
    page = tmp_path / "docs" / "page.mdx"
    page.parent.mkdir(parents=True)
    text = "import Intro from '/_includes/intro.mdx';\n\n# My Page\n\n<Intro />\n\nMore content.\n"
    result = inline_mdx_includes(text, page, tmp_path)
    assert "This is the intro." in result
    assert "<Intro" not in result
    assert "# My Page" in result
    assert "More content." in result


def test_inline_mdx_includes_nested(tmp_path):
    # Inner include
    inner = tmp_path / "_includes" / "inner.mdx"
    inner.parent.mkdir(parents=True)
    inner.write_text("Inner content here.\n")
    # Outer include that imports inner
    outer = tmp_path / "_includes" / "outer.mdx"
    outer.write_text(
        "import Inner from '/_includes/inner.mdx';\nOuter start.\n<Inner />\nOuter end.\n"
    )
    # Page that imports outer
    page = tmp_path / "page.mdx"
    text = "import Outer from '/_includes/outer.mdx';\n\n<Outer />\n"
    result = inline_mdx_includes(text, page, tmp_path)
    assert "Outer start." in result
    assert "Inner content here." in result
    assert "Outer end." in result


def test_inline_mdx_includes_missing_file(tmp_path):
    page = tmp_path / "page.mdx"
    text = "import Missing from '/_includes/missing.mdx';\n\n<Missing />\n"
    # File doesn't exist, so inline_mdx_includes leaves tag as-is
    # (strip_mdx_syntax handles cleanup later in the pipeline)
    result = inline_mdx_includes(text, page, tmp_path)
    assert result == text


def test_resolve_document(tmp_path):
    # Create a source code file with markers
    code_dir = tmp_path / "_includes" / "code"
    code_dir.mkdir(parents=True)
    source = code_dir / "example.py"
    source.write_text("# Example\nimport weaviate\n# END Example\n")
    # Create an MDX include
    intro = tmp_path / "_includes" / "intro.mdx"
    intro.write_text("Welcome to the docs.\n")
    # Create the top-level page
    page = tmp_path / "docs" / "page.mdx"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\ntitle: My Page\ntags: [search]\n---\n"
        "import Intro from '/_includes/intro.mdx';\n"
        "import PyCode from '!!raw-loader!/_includes/code/example.py';\n"
        "\n"
        "<Intro />\n"
        "\n"
        ":::tip\nUseful tip here.\n:::\n"
        "\n"
        '<FilteredTextBlock text={PyCode} startMarker="# Example" endMarker="# END Example" language="py" />\n'
    )
    doc = resolve_document(page, tmp_path)
    assert doc is not None
    assert doc.title == "My Page"
    assert doc.metadata == {"tags": ["search"]}
    assert "Welcome to the docs." in doc.body
    assert "**Tip:**" in doc.body
    assert "Useful tip here." in doc.body
    assert "```py" in doc.body
    assert "import weaviate" in doc.body
    assert "FilteredTextBlock" not in doc.body
    assert "<Intro" not in doc.body
    assert "import Intro from" not in doc.body
    assert "import PyCode from" not in doc.body
