import dataclasses
import json
from pathlib import Path

from rag.load import load_documents
from rag.models import Document


def _write_cache(cache_dir, documents):
    """Helper to write a documents cache file."""
    cache_dir.mkdir(exist_ok=True)
    documents_path = cache_dir / "documents.json"
    with open(documents_path, "w") as f:
        json.dump([dataclasses.asdict(d) for d in documents], f, default=str)


def test_loads_from_cache(tmp_path):
    docs = [Document(title="Test", body="body text", source_path=Path("docs/test.md"))]
    _write_cache(tmp_path, docs)

    result = load_documents("http://unused", Path("/unused"), tmp_path, rebuild=False)
    assert len(result) == 1
    assert result[0].title == "Test"
    assert result[0].body == "body text"
    assert result[0].source_path == Path("docs/test.md")


def test_cache_round_trips_metadata(tmp_path):
    docs = [Document(title="Test", body="body", source_path=Path("test.md"), metadata={"key": "value"})]
    _write_cache(tmp_path, docs)

    result = load_documents("http://unused", Path("/unused"), tmp_path, rebuild=False)
    assert result[0].metadata == {"key": "value"}


def test_cache_handles_missing_metadata(tmp_path):
    """Old cache files without metadata field should still load."""
    cache_dir = tmp_path
    cache_dir.mkdir(exist_ok=True)
    documents_path = cache_dir / "documents.json"
    with open(documents_path, "w") as f:
        json.dump([{"title": "Test", "body": "body", "source_path": "test.md"}], f)

    result = load_documents("http://unused", Path("/unused"), tmp_path, rebuild=False)
    assert result[0].metadata == {}


def test_returns_document_instances(tmp_path):
    docs = [Document(title="Test", body="body", source_path=Path("test.md"))]
    _write_cache(tmp_path, docs)

    result = load_documents("http://unused", Path("/unused"), tmp_path, rebuild=False)
    assert isinstance(result[0], Document)
    assert isinstance(result[0].source_path, Path)
