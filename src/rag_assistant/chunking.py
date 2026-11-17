"""Split documents into small chunks that fit nicely into a prompt.

Strategy:
1. Markdown docs are split on "## " headings first, so each chunk stays on one topic.
   Text and PDF files go straight to step 2.
2. Anything still longer than `max_chars` is split on paragraph/sentence boundaries
   with a small overlap, so a fact cut in half still shows up whole in one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loaders import Document


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str  # document title + section heading, shown as the citation
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs for each '## ' section."""
    sections: list[tuple[str, str]] = []
    current_heading, current_lines = "", []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading, current_lines = line[3:].strip(), []
        elif line.startswith("# "):
            continue  # document title, already stored on the Document
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return [(h, b) for h, b in sections if b]


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    pieces, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current.strip())
            current = current[-overlap:] if overlap else ""
        current += " " + sentence
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_documents(docs: list[Document], max_chars: int = 700, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        if doc.metadata.get("format") == "markdown":
            sections = _split_by_headings(doc.text) or [("", doc.text)]
        else:
            sections = [("", doc.text)]

        for heading, body in sections:
            for i, piece in enumerate(_split_long(body, max_chars, overlap)):
                title = f"{doc.title} > {heading}" if heading else doc.title
                # Prefix the heading so the chunk makes sense on its own (helps retrieval a lot).
                text = f"{title}\n{piece}"
                slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-") or "main"
                chunks.append(
                    Chunk(
                        id=f"{doc.id}#{slug}-{i}",
                        doc_id=doc.id,
                        title=title,
                        text=text,
                        source=doc.source,
                        metadata=dict(doc.metadata),
                    )
                )
    return chunks
