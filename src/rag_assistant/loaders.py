"""Load documents from a folder (Markdown, plain text, PDF) into plain Documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}


@dataclass
class Document:
    """A piece of source content before chunking."""

    id: str
    title: str
    text: str
    source: str  # file name, shown in citations
    metadata: dict = field(default_factory=dict)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Reading PDFs needs pypdf. Run: pip install pypdf") from exc
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_file(path: str | Path) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    text = _read_pdf(path) if suffix == ".pdf" else path.read_text(encoding="utf-8")
    first_heading = re.search(r"^#\s+(.+)$", text, flags=re.M)
    title = first_heading.group(1).strip() if first_heading else path.stem.replace("-", " ").replace("_", " ").title()
    return Document(
        id=f"doc-{path.stem}",
        title=title,
        text=text,
        source=path.name,
        metadata={"format": "markdown" if suffix in {".md", ".markdown"} else suffix.lstrip(".")},
    )


def load_folder(folder: str | Path) -> list[Document]:
    """Load every supported file in a folder (and its subfolders)."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    docs = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            doc = load_file(path)
            if doc.text.strip():
                docs.append(doc)
    return docs
