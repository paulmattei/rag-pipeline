# Architecture

Comprehensive architecture document for the Weaviate-to-Weaviate RAG pipeline. Updated March 2026.

---

## Table of Contents

1. [System Purpose](#1-system-purpose)
2. [Current State](#2-current-state)
3. [Data Flow](#3-data-flow)
4. [Module Reference](#4-module-reference)
5. [Data Structures](#5-data-structures)
6. [Chunking Strategies](#6-chunking-strategies)
7. [Evaluation System](#7-evaluation-system)
8. [Caching Strategy](#8-caching-strategy)
9. [Import Pattern Coverage](#9-import-pattern-coverage)
10. [Regex Inventory](#10-regex-inventory)
11. [Known Issues & Improvement Areas](#11-known-issues--improvement-areas)
12. [Open Questions](#12-open-questions)

---

## 1. System Purpose

Take the Weaviate documentation (MDX/MD files in a Docusaurus repo on GitHub), resolve all cross-file references into self-contained documents, chunk them, embed them, store them in a Weaviate vector database, and answer questions using Claude with retrieved context. The baseline comparison is Weaviate's own site search (powered by Weaviate + Kapa.ai).

```
GitHub Repo ──> Parse & Resolve ──> Chunk ──> Embed ──> Store in Weaviate ──> Retrieve ──> Generate
(weaviate-io)   (MDX/MD files)      (text)    (vectors)  (embedded Weaviate)   (query)      (Claude)
     [done]          [done]          [done]     [done]        [done]            [done]       [done]
```

---

## 2. Current State

### 2.1 Project Layout

```
rag/
├── src/rag/                    # Package (15 modules, ~1590 lines)
│   ├── __init__.py
│   ├── models.py               # Document, ImportGraph, Chunk dataclasses
│   ├── git.py                  # clone_or_pull (repo cloning)
│   ├── parsing.py              # MDX parsing, resolution, import graph building (~350 lines)
│   ├── load.py                 # Document loading & JSON caching
│   ├── config.py               # All pipeline configuration constants
│   ├── chunking.py             # 6 chunking strategies (~260 lines)
│   ├── embedding.py            # Local (sentence-transformers) & API (litellm) embeddings
│   ├── retrieval.py            # Weaviate batch indexing
│   ├── reranker.py             # Cross-encoder reranking
│   ├── pipeline.py             # Pipeline orchestration (~220 lines)
│   ├── llm.py                  # Claude generation + HYDE
│   ├── llm_judge.py            # LLM-based fact extraction & recall scoring (~150 lines)
│   ├── eval.py                 # Source matching + evaluation dispatch
│   └── sheets.py               # Google Sheets benchmark I/O (~275 lines)
├── main.py                     # Entry point
├── tests/                      # 82 tests across 5 files
│   ├── test_import_docs.py     # 41 tests — parsing & resolution
│   ├── test_chunking.py        # 27 tests — chunking strategies
│   ├── test_eval.py            # 6 tests — evaluation helpers
│   ├── test_load.py            # 4 tests — document loading
│   └── test_reranker.py        # 4 tests — reranker
├── misc/
│   └── serve_md_docs.py        # Local HTML preview server
├── confluence/                 # Bruno API collection for Confluence REST API exploration
├── search.py                   # Ad-hoc search script
├── diagnose_missing.py         # Debugging: find missing expected documents
├── diagnose_ranking.py         # Debugging: analyze retrieval ranking
├── chunks_anaysis.py           # Debugging: chunk size distribution analysis
├── pyproject.toml              # Project config (setuptools, ruff, dependencies)
├── cache/                      # Cached documents, chunks, embeddings (gitignored)
└── weaviate-docs/              # Cloned Weaviate docs repo (source data)
```

### 2.2 Dependencies

**Core:** `pyyaml`, `python-dotenv`, `requests`, `markdown`
**ML/Embedding:** `sentence-transformers`, `optimum[onnxruntime]`, `onnxruntime`, `litellm`
**Database:** `weaviate-client`
**Google Sheets:** `google-api-python-client`, `google-auth-oauthlib`, `google-auth-httplib2`
**Visualization:** `matplotlib`
**Dev:** `pytest`, `ruff`

### 2.3 What the Weaviate Docs Look Like

329 markdown/MDX files across these directories:
- `blog/` (~203 files) — dated directories each containing `index.mdx`
- `src/pages/` (17 files) — standalone pages (SLA, DPA, learning center)
- `_includes/` (14 MDX + 18 code files) — reusable fragments and code examples
- `papers/` (35 files) — research paper summaries
- `playbook/` (20 files) — operational guides
- `apple-and-weaviate/` (5 files) — case study

Two import patterns dominate:

**Pattern 1 — MDX includes:** A page imports another `.mdx` file and renders it as `<ComponentName />`. The included file may itself import further includes, up to 3 levels deep.

**Pattern 2 — FilteredTextBlock:** A page imports a raw code file via `!!raw-loader!` and uses `<FilteredTextBlock text={VarName} startMarker="..." endMarker="..." />` to extract a section between comment markers.

---

## 3. Data Flow

### 3.1 End-to-End Pipeline

```
1. clone_or_pull()
   └── git clone --depth 1 (skips if repo exists)

2. load_documents(repo_url, repo_dir, cache_dir)
   ├── If cache/documents.json exists → deserialize and return
   └── Otherwise:
       ├── build_import_graph(repo_dir)
       │   ├── rglob("*") for .md/.mdx files           → 329 files
       │   ├── parse_file_imports() on each             → resolve import paths
       │   └── return ImportGraph                        → {file: [dependencies]}
       ├── graph.top_level_pages()
       │   └── files not imported by others, excluding _includes/
       └── For each top-level page:
           resolve_document(page_path, repo_dir)
           ├── parse_document()                         → Document with raw body
           ├── inline_mdx_includes()                    → replace <Component /> with file content
           │   ├── parse_mdx_imports()                  → {name: Path}
           │   ├── read each imported file
           │   ├── strip_frontmatter() on included content
           │   └── recurse (max depth 3 from config)
           ├── resolve_filtered_text_blocks()            → replace <FilteredTextBlock /> with code
           │   ├── parse_raw_imports()                  → {var: path}
           │   ├── resolve_import_path()                → absolute Path
           │   ├── read source file
           │   └── extract_between_markers()            → pull code section
           └── strip_mdx_syntax()                       → remove JSX, convert admonitions

3. read_benchmark(spreadsheet_id)
   └── Fetch expected sources + benchmark answers from Google Sheets

4. load_key_facts(benchmark_answers)
   ├── If cache/key_facts.json exists → load
   └── Otherwise: LLM extracts key facts from each benchmark answer, caches

5. For each (strategy, embedding_model, retrieval_k, reranker_model):
   ├── setup_collection(client, documents, strategy, embedding_model)
   │   ├── load_chunks(documents, strategy, cache_dir)
   │   │   ├── If cache/{strategy}/chunks.json exists → load
   │   │   └── Otherwise: create_chunks() per document, cache
   │   ├── Filter oversized chunks (>8191 tokens for API models)
   │   ├── load_embeddings(chunks, strategy, embedding_model, cache_dir)
   │   │   ├── If cache/{strategy}/{model}/embeddings.npy exists → load
   │   │   └── Otherwise: embed_chunks(), save as .npy
   │   ├── Create Weaviate collection (vectorizer=none)
   │   └── index_chunks() → batch insert with embeddings
   │
   └── run_queries(collection, expected_sources, key_facts_by_query, ...)
       └── For each of 12 benchmark queries:
           process_query()
           ├── Embed query (or generate HYDE hypothetical document first)
           ├── Search: near_vector or hybrid (BM25 + vector fusion)
           ├── Rerank top results with cross-encoder (if configured)
           ├── Build LLM context (deduplicated parent_text or child text)
           ├── generate_response() → Claude Sonnet 4.6
           └── evaluate() → source matching + LLM-judged fact recall

6. update_summary_tab(spreadsheet_id, all_results, key_facts_by_query)
   └── Write multi-dimensional grid to Google Sheets Summary tab
```

### 3.2 What Happens to a Single Page

Take `blog/2023-05-05-generative-feedback-loops/index.mdx`:

```
Raw file (85 lines)
  │
  ├── Frontmatter extracted: {title: "Generative Feedback Loops", tags: [...]}
  ├── Body: remaining ~75 lines of MDX
  │
  ├── inline_mdx_includes():
  │   ├── import GenerativeSearch from '/_includes/code/generative.feedback.loops.search.mdx'
  │   │   └── That file contains <Tabs>/<TabItem> with code blocks → inlined
  │   ├── import GenerativeDescription from '/_includes/code/generative.feedback.loops.description.mdx'
  │   │   └── inlined
  │   └── import GenerativeLoop from '/_includes/code/generative.feedback.loops.loop.mdx'
  │       └── inlined
  │
  ├── resolve_filtered_text_blocks():
  │   └── (if any FilteredTextBlock tags exist after inlining)
  │
  └── strip_mdx_syntax():
      ├── Remove: import statements, HTML comments, JSX tags
      ├── Convert: :::tip → **Tip:**, <TabItem label="Python"> → **Python:**
      └── Clean: collapse blank lines

Result: ~200 lines of clean markdown with all code examples inlined
```

---

## 4. Module Reference

### 4.1 Dependency Graph

```
main.py
├── rag.config          (constants)
├── rag.load            (document loading & caching)
│   ├── rag.git         (clone_or_pull)
│   └── rag.parsing     (build_import_graph, resolve_document)
│       ├── rag.config  (MAX_INCLUDE_DEPTH)
│       └── rag.models  (Document, ImportGraph)
├── rag.sheets          (read_benchmark, update_summary_tab)
├── rag.llm_judge       (extract_key_facts)
└── rag.pipeline        (setup_collection, run_queries)
    ├── rag.chunking    (create_chunks)
    │   └── rag.models  (Chunk)
    ├── rag.embedding   (embed_chunks)
    ├── rag.retrieval   (index_chunks)
    ├── rag.reranker    (rerank)
    ├── rag.llm         (generate_response, generate_hyde_document)
    ├── rag.eval        (evaluate)
    │   └── rag.llm_judge (check_retrieval, score_recall)
    └── rag.sheets      (create_run_tab, format_result_row)
```

### 4.2 Function Index

```
git.py:
  clone_or_pull(repo_url, local_path)                              # Shallow clone; no-op if exists

parsing.py:
  extract_frontmatter(text) -> dict                                # YAML frontmatter → dict
  strip_frontmatter(text) -> str                                   # Remove frontmatter block
  extract_between_markers(text, start, end) -> str                 # Code between comment markers
  strip_mdx_syntax(text) -> str                                    # Remove JSX, convert admonitions
  resolve_import_path(import_path, file_path, repo_dir) -> Path    # /, ./, ../ path resolution
  parse_file_imports(text, file_path, repo_dir) -> [Path]          # All imports → Paths (for graph)
  parse_raw_imports(text) -> {var: path}                           # !!raw-loader! imports (for FilteredTextBlock)
  parse_mdx_imports(text, file_path, repo_dir) -> {name: Path}    # MDX imports (for inlining)
  parse_document_text(text, file_path) -> Document                 # Text → Document (no I/O)
  parse_document(file_path) -> Document | None                     # File → Document
  resolve_filtered_text_blocks(text, file_path, repo_dir) -> str   # <FilteredTextBlock /> → code blocks
  inline_mdx_includes(text, file_path, repo_dir, _depth) -> str   # <Component /> → file content
  build_import_graph(repo_dir) -> ImportGraph                      # Full repo dependency scan
  resolve_document(page_path, repo_dir) -> Document | None         # Full resolution pipeline

load.py:
  resolve_all_documents(repo_dir) -> [Document]                    # Graph → resolve all top-level pages
  load_documents(repo_url, repo_dir, cache_dir, rebuild) -> [Document]  # Load with JSON caching

config.py:
  STRATEGIES, EMBEDDING_MODELS, RERANKER_MODELS                    # Evaluation sweep axes
  RETRIEVAL_K_VALUES, TOP_K, SEARCH_TYPE, HYBRID_ALPHA             # Retrieval config
  QUERY_TRANSFORM, QUERIES, SYSTEM_PROMPT                          # Query/LLM config
  MAX_INCLUDE_DEPTH, CACHE_DIR, SPREADSHEET_ID                     # Infrastructure

chunking.py:
  create_chunks(text, title, source_path, strategy) -> [Chunk]     # Strategy dispatcher
  create_chunks_500_char(...)                                      # Fixed 500-char chunks
  create_chunks_500_char_with_runt_check(...)                      # 500-char + merge trailing runt
  create_chunks_document(...)                                      # Whole document as one chunk
  create_chunks_markdown_sections(...)                             # Split on headers + breadcrumbs
  create_chunks_markdown_optimized(...)                            # Header-aware + parent-child + size cap
  header_level(line) -> int                                        # Detect markdown header
  split_by_headers(text) -> [(header, body)]                       # Split at header boundaries
  build_breadcrumbs(sections) -> [str]                             # Ancestor header trail
  split_at_paragraphs(text, max_chars) -> [str]                    # Split at paragraph boundaries
  source_category(source_path) -> str                              # Extract docs directory category
  format_chunk_text(title, breadcrumb, body, category) -> str      # Build embeddable chunk text
  _hard_split_text(text, max_chars) -> [str]                       # Word-boundary hard split

embedding.py:
  embed_chunks(texts, model) -> np.ndarray                         # Local or API embedding
  _get_local_model(model) -> SentenceTransformer                   # Lazy-load on MPS device

retrieval.py:
  index_chunks(collection, chunks, embeddings)                     # Batch insert into Weaviate

reranker.py:
  rerank(query, texts, model, top_n) -> [int]                      # Cross-encoder rerank → indices
  _get_model(model) -> CrossEncoder                                # Lazy-load on MPS device

pipeline.py:
  load_chunks(documents, strategy, cache_dir, rebuild) -> [Chunk]           # Chunk with JSON caching
  load_embeddings(chunks, strategy, embedding_model, cache_dir, rebuild)    # Embed with .npy caching
  setup_collection(client, documents, strategy, embedding_model, rebuild)   # Full index setup
  process_query(collection, query, expected, key_facts, ...)                # Single query end-to-end
  run_queries(collection, expected_sources, key_facts, ...)                 # All 12 queries for a config
  run_label(strategy, embedding_model, retrieval_k, reranker_model) -> str  # Human-readable label
  _build_llm_context(objects) -> str                               # Deduplicated parent text for LLM
  _filter_oversized_chunks(chunks, max_tokens) -> [Chunk]          # Token-limit filter
  _collection_name(strategy, embedding_model) -> str               # Weaviate collection name

llm.py:
  generate_response(prompt) -> (str, int, int)                     # Claude Sonnet 4.6 generation
  generate_hyde_document(query) -> str                              # Hypothetical Document Embedding

llm_judge.py:
  extract_key_facts(query, benchmark_answer) -> [str]              # LLM extracts atomic facts
  check_retrieval(query, key_facts, chunks_text) -> [bool]         # Which facts in retrieved chunks?
  score_recall(query, key_facts, rag_answer) -> (str, [bool], str) # Which facts in final answer?
  _parse_json_response(content) -> dict                            # Robust JSON extraction from LLM

eval.py:
  evaluate(query, response, objects, expected, key_facts) -> dict  # Full evaluation for one query
  compute_expected_match(retrieved_paths, expected_paths) -> str    # Source path matching

sheets.py:
  get_credentials() -> Credentials                                 # OAuth2 with cached token.json
  get_service() -> SheetsService                                   # Build Sheets API client
  read_benchmark(spreadsheet_id) -> (dict, dict)                   # Read expected sources + answers
  format_result_row(...) -> dict                                   # Format result for spreadsheet
  create_run_tab(spreadsheet_id, run_name, results)                # Write per-run tab
  update_summary_tab(spreadsheet_id, all_results, key_facts)       # Write Summary grid tab
```

---

## 5. Data Structures

```python
@dataclass
class Document:
    title: str           # From frontmatter, falls back to filename stem
    body: str            # Clean markdown after full resolution
    source_path: Path    # Original file on disk
    metadata: dict       # Remaining frontmatter (tags, date, slug, etc.)

@dataclass
class ImportGraph:
    pages: dict[Path, list[Path]]  # file → [imported files]
    def top_level_pages(self) -> list[Path]
        # Files not imported by others, excluding _includes/ directory

@dataclass
class Chunk:
    text: str                # Chunk content (with title/breadcrumb prefix)
    document_title: str      # Source document title
    source_path: Path        # Original file path
    chunk_index: int         # Position within document
    parent_text: str = ""    # Parent section text (markdown-optimized only)
```

---

## 6. Chunking Strategies

Six strategies are implemented. The pipeline sweeps over strategies configured in `config.py`.

| Strategy | Description | Key Behavior |
|----------|-------------|-------------|
| `500-char` | Fixed 500-char chunks | No overlap, splits mid-word |
| `500-char-with-runt-check` | 500-char + merge trailing runt | Merges final chunk if <500 chars |
| `document` | Entire document = one chunk | Baseline for whole-document retrieval |
| `markdown-sections` | Split on headers + breadcrumbs | Prefix with `Title > H2 > H3` trail; split oversized sections at paragraphs |
| `markdown-sections-no-breadcrumbs` | Same without breadcrumb prefix | For comparing breadcrumb impact |
| `markdown-optimized` | Header-aware with parent-child and size cap | **Currently used.** Child chunks ~1500 chars, each stores parent section text for richer LLM context |

### markdown-optimized Details

The active strategy (`markdown-optimized`) works as follows:

1. Split document at header boundaries
2. Build breadcrumb trail per section
3. For each section:
   - Compute body budget = `max_child_chars` (1500) minus breadcrumb prefix length
   - If body fits → one child chunk
   - If body too large → split at paragraph boundaries, hard-split oversized paragraphs at word boundaries
   - Each child stores the full parent section text
4. Merge adjacent children below `min_child_chars` (200)
5. Merge trailing runt chunk backward

At query time, `_build_llm_context` sends deduplicated parent_text to Claude rather than individual child chunks, giving the LLM broader context around each retrieval hit.

---

## 7. Evaluation System

### 7.1 Benchmark Data

12 queries stored in `config.py`, with expected source paths and hand-written benchmark answers in a Google Sheet (`Benchmark` tab).

### 7.2 Evaluation Pipeline

Each query is evaluated on two dimensions:

**Source matching** (`eval.py`): Compare retrieved chunk source paths against expected paths.

**Fact recall** (`llm_judge.py`): An LLM-as-judge approach:
1. At startup, Claude extracts atomic key facts from each benchmark answer (`extract_key_facts`)
2. Per query, Claude checks which facts appear in retrieved chunks (`check_retrieval`)
3. Per query, Claude checks which facts appear in the generated answer (`score_recall`)
4. Results are per-fact booleans + a recall score like `"7/12"`

### 7.3 Reporting

Results go to Google Sheets:
- **Per-run tab**: One row per query with all metadata (chunks returned, prompt, answer, tokens, scores)
- **Summary tab**: Multi-dimensional grid — rows are queries x facts, columns are configurations with Retrieved/Answered sub-columns

### 7.4 Configuration Sweep

`main.py` loops over the Cartesian product of:
- `STRATEGIES` (chunking strategies)
- `EMBEDDING_MODELS` (embedding models)
- `RETRIEVAL_K_VALUES` (k to retrieve from vector search)
- `RERANKER_MODELS` (reranker models, `None` = no reranking)

Current defaults: `markdown-optimized`, `BAAI/bge-base-en-v1.5`, `k=250`, `BAAI/bge-reranker-base`.

### 7.5 Search Modes

- **Vector** (`SEARCH_TYPE="vector"`): Pure `near_vector` search. Currently active.
- **Hybrid** (`SEARCH_TYPE="hybrid"`): BM25 + vector fusion, controlled by `HYBRID_ALPHA` (0=BM25, 1=vector).
- **HYDE** (`QUERY_TRANSFORM="hyde"`): Generate a hypothetical document passage, embed that instead of the raw query.

---

## 8. Caching Strategy

Each pipeline stage caches independently, enabling rapid iteration on downstream stages without re-running upstream ones.

```
cache/
├── documents.json                          # Resolved documents (load.py)
├── key_facts.json                          # LLM-extracted key facts (main.py)
└── {strategy}/
    ├── chunks.json                         # Chunked documents (pipeline.py)
    └── {model-slug}/
        └── embeddings.npy                  # Embeddings as numpy array (pipeline.py)
```

`--rebuild` flag forces all caches to be regenerated.

---

## 9. Import Pattern Coverage

Patterns found in the Weaviate docs repo and whether the current parsers handle them:

| Pattern | Example | Handled? | Parser |
|---------|---------|----------|--------|
| Default import, absolute path | `import Intro from '/_includes/intro.mdx'` | Yes | `parse_file_imports`, `parse_mdx_imports` |
| Default import, relative `./` | `import Local from './_local.mdx'` | Yes | `parse_file_imports`, `parse_mdx_imports` |
| Default import, relative `../` | `import Config from '../../config.ts'` | Yes | All parsers via `resolve_import_path` |
| Raw-loader import | `import Code from '!!raw-loader!/_includes/code.py'` | Yes | `parse_file_imports`, `parse_raw_imports` |
| Named import `{ X }` | `import { MetaSEO } from '/src/theme/MetaSEO'` | Partial | `strip_mdx_syntax` removes it, but `parse_file_imports` skips it |
| Theme import | `import Tabs from '@theme/Tabs'` | Correctly skipped | All parsers |
| Package import | `import weaviate from 'weaviate-client'` | Correctly skipped | All parsers |
| Asset import (video/image) | `import demo from './img/demo.mp4'` | Skipped (no extension match) | `parse_file_imports` |
| `require()` in JSX | `src={require('./img/dag.png').default}` | Not parsed | Stripped by JSX tag removal |

---

## 10. Regex Inventory

Every regex in the codebase, its purpose, and known issues:

| Location | Regex | Purpose | Issue |
|----------|-------|---------|-------|
| `parsing.py` `_FRONTMATTER_RE` | `\A---\n(.*?\n)?---\n?` | Match YAML frontmatter block | Fails on CRLF line endings |
| `parse_file_imports` | `^import\s+\w+\s+from\s+['\"](?:!!raw-loader!)?(.+?\.(?:mdx?|py|ts|js|go|java))['\"]` | Match file imports with extensions | Skips named imports `{ X }` |
| `parse_raw_imports` | `^import\s+(\w+)\s+from\s+['\"]!!raw-loader!(.+?)['\"]` | Match raw-loader imports | None known |
| `parse_mdx_imports` | `^import\s+(\w+)\s+from\s+['\"]([^'\"!@]+\.mdx?)['\"]\s*;?\s*$` | Match MDX file imports | None known |
| `resolve_filtered_text_blocks` | `<FilteredTextBlock\s+(.*?)\s*/>` | Match self-closing FilteredTextBlock | DOTALL — correct |
| `resolve_filtered_text_blocks` (inner) | `text=\{(\w+)\}`, `startMarker="(.+?)"`, etc. | Extract FilteredTextBlock props | None known |
| `strip_mdx_syntax` (code split) | `` (^(?:`{3,}|~{3,}).*$\n(?:.*\n)*?^(?:`{3,}|~{3,})\s*$) `` | Separate code blocks from non-code | None known |
| `strip_mdx_syntax` (imports) | `^import\s+(?:\w+|\{[^}]+\})\s+from\s+['\"].*['\"]\s*;?\s*$\n?` | Remove JS/MDX imports | Only applied to non-code segments |
| `strip_mdx_syntax` (comments) | `<!--.*?-->` | Remove HTML comments | None known |
| `strip_mdx_syntax` (admonitions) | `^:::(tip|note|warning|caution|danger|info)(?:[ \t]+(.+))?$` | Convert admonition openers | None known |
| `strip_mdx_syntax` (admonition close) | `^:::$\n?` | Remove closing `:::` | None known |
| `strip_mdx_syntax` (TabItem) | `<TabItem\s[^>]*label="([^"]+)"[^>]*>` | Extract tab labels | None known |
| `strip_mdx_syntax` (JSX tags) | `</?[A-Za-z][^>]*>` | Remove all HTML/JSX tags | Also removes legitimate HTML (`<details>`, `<summary>`) |
| `strip_mdx_syntax` (blank lines) | `\n{3,}` | Collapse excessive whitespace | None known |
| `inline_mdx_includes` | `<{name}\s*/>` (dynamic) | Replace component usage | `content` used as replacement could contain `\1` backreference patterns |

---

## 11. Known Issues & Improvement Areas

### 11.1 The Import Graph Is Built but Not Used for Resolution

`build_import_graph` scans all 329 files and discovers dependencies. Then `resolve_document` re-discovers dependencies per page:

```python
text = inline_mdx_includes(text, page_path, repo_dir)      # re-parses imports
text = resolve_filtered_text_blocks(text, page_path, ...)   # re-parses raw-loader imports
```

Import parsing happens 3 times per file (once in graph, once for MDX, once for raw-loader). The same included file is read from disk N times if N pages import it.

A graph-driven resolution engine — topological sort the graph, resolve bottom-up with caching — would read every file exactly once. Not currently a bottleneck (the whole resolve phase is a few seconds), but would be cleaner.

### 11.2 No Cycle Detection in `inline_mdx_includes`

Recursion is depth-limited to `MAX_INCLUDE_DEPTH` (3), which prevents stack overflow. But there is no explicit cycle detection — a circular import would silently stop inlining at the depth limit rather than reporting an error.

### 11.3 `build_import_graph` Scans the Entire Repo

`rglob("*")` traverses all directories including `node_modules/`, `.docusaurus/`, etc. Could be restricted to known content directories.

### 11.4 `extract_between_markers` Collects to EOF on Missing End Marker

If a source file has a start marker but no end marker, the function collects everything to EOF. This is documented and tested, but is a silent data quality issue.

### 11.5 No Path Traversal Protection

`resolve_import_path` resolves paths without verifying they stay within `repo_dir`. Low risk (we control the input) but is a bad pattern.

### 11.6 `clone_or_pull` Doesn't Update

The pull logic is commented out — the function is currently a no-op if the repo exists. To pick up doc changes, the repo must be deleted and re-cloned.

### 11.7 No Parallelism in Resolution

Top-level pages are resolved sequentially. Each `resolve_document` does multiple file reads. A thread pool would help since the work is I/O-bound, though caching (11.1) would help more.

---

## 12. Open Questions

1. **Incremental updates.** When the Weaviate docs change, do we re-process everything or detect changes? `git diff` could identify changed files, but the import graph means a change to `_includes/intro.mdx` invalidates every page that imports it.

2. **Code example versioning.** The docs have Python v3/v4 client code, TypeScript v2/v3, etc. Should chunks for different versions be separate? Should the retriever prefer the latest version?

3. **Duplicate content.** Some pages share significant content via includes. After inlining, chunks from different pages may be near-identical. Should we deduplicate at the chunk level?

4. **Evaluation dataset size.** 12 benchmark queries is enough to iterate on pipeline configuration, but may not be sufficient to draw confident conclusions about which strategies generalize.
