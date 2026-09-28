"""Document loading, chunking, embedding, and ChromaDB indexing."""

from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass(frozen=True)
class PageRecord:
    """One page (or whole file) of extracted text with provenance."""

    text: str
    page_number: int
    source_file: str


def load_pdf(file_bytes: bytes, filename: str) -> list[PageRecord]:
    """Extract text from a PDF, one record per page (1-indexed page numbers)."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not read PDF '{filename}': {exc}") from exc

    if len(reader.pages) == 0:
        raise ValueError(f"PDF '{filename}' has no pages.")

    pages: list[PageRecord] = []
    for index, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        pages.append(
            PageRecord(
                text=raw.strip(),
                page_number=index + 1,
                source_file=filename,
            )
        )
    return pages


def load_text_or_markdown(content: str, filename: str) -> list[PageRecord]:
    """Treat a .txt or .md file as a single logical page."""
    text = content.strip()
    if not text:
        raise ValueError(f"File '{filename}' is empty.")
    return [
        PageRecord(text=text, page_number=1, source_file=filename),
    ]


def load_upload(filename: str, file_bytes: bytes) -> list[PageRecord]:
    """Dispatch loader by file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return load_pdf(file_bytes, filename)
    if lower.endswith((".txt", ".md")):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File '{filename}' must be UTF-8 text.") from exc
        return load_text_or_markdown(text, filename)
    raise ValueError(
        f"Unsupported file type for '{filename}'. Use PDF, .txt, or .md."
    )


@dataclass(frozen=True)
class TextChunk:
    """A slice of document text with citation metadata."""

    text: str
    source_file: str
    page_number: int
    chunk_index: int


def chunk_pages(
    pages: list[PageRecord],
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    """Split page text into overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size.")

    chunks: list[TextChunk] = []
    global_index = 0

    for page in pages:
        text = page.text
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            slice_text = text[start:end].strip()
            if slice_text:
                chunks.append(
                    TextChunk(
                        text=slice_text,
                        source_file=page.source_file,
                        page_number=page.page_number,
                        chunk_index=global_index,
                    )
                )
                global_index += 1
            if end >= len(text):
                break
            start = end - overlap

    return chunks
