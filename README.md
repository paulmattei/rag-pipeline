# Weaviate Docs RAG Pipeline

End-to-end RAG pipeline that takes the Weaviate documentation (MDX/MD files from GitHub), resolves cross-file references into self-contained documents, chunks them, embeds them, stores them in Weaviate, and answers questions using Claude. Evaluation uses an LLM-as-judge approach with results reported to Google Sheets.

## Pipeline

```
GitHub Repo ──> Parse & Resolve ──> Chunk ──> Embed ──> Store in Weaviate ──> Retrieve ──> Generate
(weaviate-io)   (MDX/MD files)      (text)    (vectors)  (embedded Weaviate)   (query)      (Claude)
```

## Quick Start

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env  # then fill in REPO_URL, REPO_DIR, ATLASSIAN_API_KEY

# Run the full pipeline
uv run python main.py

# Force rebuild all caches
uv run python main.py --rebuild

# Run tests
uv run pytest tests/ -v

# Lint and format
uv run ruff check .
uv run ruff format .
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `REPO_URL` | GitHub URL for the Weaviate docs repo |
| `REPO_DIR` | Local directory for the cloned repo |
| `ANTHROPIC_API_KEY` | For Claude generation and LLM-judge evaluation |
| `SPREADSHEET_ID` | Google Sheets ID for benchmark data and run results |

## How It Works

### 1. Parse & Resolve

The Weaviate docs use Docusaurus (MDX). Pages import content from other files up to 3 levels deep. Two patterns dominate:

- **MDX includes** -- `import Intro from './_intro.mdx'` rendered as `<Intro />`
- **FilteredTextBlock** -- `!!raw-loader!` imports that extract code between comment markers

The pipeline builds an import graph, identifies top-level pages (not imported by others), then resolves each page by inlining includes, resolving code blocks, and stripping JSX syntax. Result: 329 source files become clean, self-contained markdown documents.

### 2. Chunk

Six chunking strategies are implemented. The active strategy is `markdown-optimized`:

1. Split at header boundaries with breadcrumb trails (`Title > H2 > H3`)
2. Child chunks target ~1500 chars, split at paragraph boundaries
3. Each child stores the full parent section text for richer LLM context
4. Small adjacent children are merged; trailing runts are merged backward

### 3. Embed & Store

Chunks are embedded using a local sentence-transformers model (`BAAI/bge-base-en-v1.5` on MPS) and batch-inserted into an embedded Weaviate instance with `vectorizer=none`.

### 4. Retrieve & Generate

Each query is embedded (or optionally transformed via HYDE), searched via `near_vector` or hybrid (BM25 + vector fusion), optionally reranked with a cross-encoder, then passed to Claude Sonnet for generation. The LLM receives deduplicated parent-section text rather than individual child chunks.

### 5. Evaluate

Each query is scored on two dimensions:

- **Source matching** -- do retrieved chunk paths match expected source documents?
- **Fact recall** (LLM-as-judge) -- Claude extracts atomic key facts from benchmark answers, then checks which facts appear in retrieved chunks and in the generated answer

Results go to Google Sheets: a per-run tab with full metadata and a summary grid comparing configurations.

## Configuration Sweep

`main.py` loops over the Cartesian product of axes defined in `config.py`:

| Axis | Current Default |
|------|----------------|
| Chunking strategy | `markdown-optimized` |
| Embedding model | `BAAI/bge-base-en-v1.5` |
| Retrieval k | `250` |
| Reranker model | `BAAI/bge-reranker-base` |
| Search type | `vector` |
| Query transform | `none` |

12 benchmark queries are evaluated per configuration.

## Caching

Each pipeline stage caches independently under `cache/`:

```
cache/
├── documents.json                    # Resolved documents
├── key_facts.json                    # LLM-extracted key facts
└── {strategy}/
    ├── chunks.json                   # Chunked documents
    └── {model-slug}/
        └── embeddings.npy            # Embeddings as numpy array
```

Use `--rebuild` to force regeneration.

## Project Structure

```
rag/
├── src/rag/                    # Package (15 modules, ~1590 lines)
│   ├── models.py               # Document, ImportGraph, Chunk dataclasses
│   ├── git.py                  # clone_or_pull (repo cloning)
│   ├── parsing.py              # MDX parsing, resolution, import graph building
│   ├── load.py                 # Document loading & JSON caching
│   ├── config.py               # All pipeline configuration constants
│   ├── chunking.py             # 6 chunking strategies
│   ├── embedding.py            # Local (sentence-transformers) & API (litellm) embeddings
│   ├── retrieval.py            # Weaviate batch indexing
│   ├── reranker.py             # Cross-encoder reranking
│   ├── pipeline.py             # Pipeline orchestration
│   ├── llm.py                  # Claude generation + HYDE
│   ├── llm_judge.py            # LLM-based fact extraction & recall scoring
│   ├── eval.py                 # Source matching + evaluation dispatch
│   └── sheets.py               # Google Sheets benchmark I/O
├── main.py                     # Entry point
├── tests/                      # 82 tests across 5 files
├── cache/                      # Cached artifacts (gitignored)
└── weaviate-docs/              # Cloned Weaviate docs repo
```

## Dependencies

**Core:** `pyyaml`, `python-dotenv`, `requests`, `markdown`
**ML/Embedding:** `sentence-transformers`, `optimum[onnxruntime]`, `onnxruntime`, `litellm`
**Database:** `weaviate-client`
**Google Sheets:** `google-api-python-client`, `google-auth-oauthlib`, `google-auth-httplib2`
**Visualization:** `matplotlib`
**Dev:** `pytest`, `ruff`

## Why Three Import Parsers?

The codebase has three functions that parse import statements, each serving a different consumer:

| | `parse_file_imports` | `parse_raw_imports` | `parse_mdx_imports` |
|---|---|---|---|
| **Used by** | `build_import_graph` | `resolve_filtered_text_blocks` | `inline_mdx_includes` |
| **Returns** | `list[Path]` | `dict[str, str]` (var -> path) | `dict[str, Path]` (name -> path) |
| **Matches** | All file imports | Only `!!raw-loader!` | Only `.md`/`.mdx` files |

See [architecture.md](architecture.md) for the full module reference, data flow diagrams, regex inventory, and known issues.
