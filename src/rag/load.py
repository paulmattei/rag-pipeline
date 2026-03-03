import dataclasses
import json
import logging
from pathlib import Path

from rag.git import clone_or_pull
from rag.models import Document
from rag.parsing import build_import_graph, resolve_document

logger = logging.getLogger(__name__)


def resolve_all_documents(repo_dir):
    """Build import graph and resolve all top-level pages.

    Returns list[Document].
    """
    graph = build_import_graph(repo_dir)
    top_level = graph.top_level_pages()
    logger.info(f"Resolving {len(top_level)} top-level pages")

    documents = []
    for page_path in top_level:
        doc = resolve_document(page_path, repo_dir)
        if doc is not None:
            documents.append(doc)

    return documents


def load_documents(repo_url, repo_dir, cache_dir, rebuild=False):
    """Load documents from cache, or clone/pull and resolve if needed."""
    documents_path = cache_dir / "documents.json"
    cache_dir.mkdir(exist_ok=True)

    if documents_path.exists() and not rebuild:
        logger.info("Loading cached documents")
        with open(documents_path) as f:
            raw = json.load(f)
        return [Document(title=d["title"], body=d["body"], source_path=Path(d["source_path"]), metadata=d.get("metadata", {})) for d in raw]

    clone_or_pull(repo_url, repo_dir)
    documents = resolve_all_documents(repo_dir)

    with open(documents_path, "w") as f:
        json.dump([dataclasses.asdict(d) for d in documents], f, default=str)

    return documents
