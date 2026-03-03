from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    title: str
    body: str
    source_path: Path
    metadata: dict = field(default_factory=dict)


@dataclass
class ImportGraph:
    pages: dict[Path, list[Path]] = field(default_factory=dict)

    def top_level_pages(self) -> list[Path]:
        """Return pages that are not imported by any other page.

        Excludes _includes/ files, which are partials meant to be inlined
        into other documents, not standalone pages.
        """
        all_imports = {p for deps in self.pages.values() for p in deps}
        return sorted(
            p for p in self.pages
            if p not in all_imports and "/_includes/" not in str(p)
        )

@dataclass
class Chunk:
    text: str
    document_title: str
    source_path: Path
    chunk_index: int
    parent_text: str = ""
