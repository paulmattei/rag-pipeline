import logging
import os
import re
import subprocess
import sys
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

import dotenv
from git import clone_or_pull
from models import Document
from parsing import build_import_graph, resolve_document

logger = logging.getLogger(__name__)


def build_url_map(documents: list[Document], repo_dir: Path) -> dict[str, Document]:
    """Map URL slugs to documents for resolving internal links.

    Uses the 'slug' from frontmatter metadata if present, otherwise
    derives the URL from the file path (mimicking Docusaurus behavior).
    Returns {url_path: Document}.
    """
    repo_dir = repo_dir.resolve()
    url_map = {}
    for doc in documents:
        rel = doc.source_path.relative_to(repo_dir)
        # Strip index.md(x) from the end
        if rel.name.startswith("index."):
            rel = rel.parent
        else:
            rel = rel.with_suffix("")
        # Strip src/pages prefix (Docusaurus convention)
        rel_str = str(rel)
        if rel_str.startswith("src/pages/"):
            rel_str = rel_str[len("src/pages/"):]
        # Strip date prefix from blog dirs (2024-01-23-title → title)
        parts = Path(rel_str).parts
        if len(parts) >= 2 and parts[0] == "blog":
            stripped = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", parts[1])
            if stripped != parts[1]:
                parts = (parts[0], stripped) + parts[2:]
                rel_str = str(Path(*parts))
        # If slug exists in metadata, replace the last path segment
        slug = doc.metadata.get("slug")
        if slug:
            parent = str(Path(rel_str).parent)
            if parent == ".":
                rel_str = slug
            else:
                rel_str = f"{parent}/{slug}"
        url = "/" + rel_str
        url_map[url] = doc
    return url_map


def serve_documents(
    documents: list[Document], url_map: dict[str, Document], port: int = 8000
):
    """Serve resolved documents as a browsable website."""
    path_to_doc = {path: doc for path, doc in url_map.items()}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = unquote(self.path).rstrip("/") or "/"
            if path == "/":
                self._serve_index()
            elif path in path_to_doc:
                self._serve_doc(path_to_doc[path])
            else:
                self.send_error(404)

        def _serve_index(self):
            items = ""
            for url in sorted(path_to_doc):
                doc = path_to_doc[url]
                items += f'<li><a href="{escape(url)}">{escape(doc.title)}</a> <small>({len(doc.body)} chars)</small></li>\n'
            html = f"<html><head><title>Weaviate Docs</title></head><body><h1>Weaviate Docs ({len(path_to_doc)} pages)</h1><ul>{items}</ul></body></html>"
            self._respond(html)

        def _serve_doc(self, doc):
            html = f"<html><head><title>{escape(doc.title)}</title></head><body><p><a href='/'>&larr; Index</a></p><h1>{escape(doc.title)}</h1><pre>{escape(doc.body)}</pre></body></html>"
            self._respond(html)

        def _respond(self, html):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())

    server = HTTPServer(("", port), Handler)
    logger.info(f"Serving {len(path_to_doc)} docs at http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dotenv.load_dotenv()
    REPO_URL = os.environ["REPO_URL"]
    REPO_DIR = Path(os.environ["REPO_DIR"])

    try:
        clone_or_pull(REPO_URL, REPO_DIR)
    except subprocess.CalledProcessError:
        sys.exit(1)
    graph = build_import_graph(REPO_DIR)
    top_level = graph.top_level_pages()
    logger.info(f"Resolving {len(top_level)} top-level pages")

    documents = []
    for page_path in top_level:
        doc = resolve_document(page_path, REPO_DIR)
        if doc is not None:
            documents.append(doc)

    logger.info(f"Resolved {len(documents)} documents")
    url_map = build_url_map(documents, REPO_DIR)
    serve_documents(documents, url_map)
