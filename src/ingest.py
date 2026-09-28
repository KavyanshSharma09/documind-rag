"""Document loading, chunking, embedding, and ChromaDB indexing."""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from pypdf import PdfReader

if TYPE_CHECKING:
    import chromadb
    from chromadb.api.models.Collection import Collection
    from sentence_transformers import SentenceTransformer

from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL


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


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent Chroma client, creating the directory if needed."""
    import chromadb

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def get_or_create_collection(client: chromadb.PersistentClient) -> Collection:
    """Get or create the DocuMind document collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(
    embedder: SentenceTransformer,
    texts: list[str],
) -> list[list[float]]:
    """Embed a batch of strings."""
    vectors = embedder.encode(texts, show_progress_bar=False)
    return vectors.tolist()


def index_chunks(
    collection: Collection,
    chunks: list[TextChunk],
    embedder: SentenceTransformer,
) -> int:
    """Embed and upsert chunks into Chroma. Returns number of chunks indexed."""
    if not chunks:
        return 0

    ids = [f"{c.source_file}:{c.chunk_index}:{uuid.uuid4().hex[:8]}" for c in chunks]
    documents = [c.text for c in chunks]
    embeddings = embed_texts(embedder, documents)
    metadatas = [
        {
            "source_file": c.source_file,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def clear_index(client: chromadb.PersistentClient) -> Collection:
    """Delete and recreate the document collection."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass
    return get_or_create_collection(client)


def ingest_uploads(
    files: list[tuple[str, bytes]],
    chunk_size: int,
    overlap: int,
    embedder: SentenceTransformer,
    collection: Collection,
) -> int:
    """Load, chunk, and index uploaded files. Returns total chunks added."""
    total = 0
    for filename, data in files:
        pages = load_upload(filename, data)
        chunks = chunk_pages(pages, chunk_size, overlap)
        if not chunks:
            raise ValueError(f"No text extracted from '{filename}'.")
        total += index_chunks(collection, chunks, embedder)
    return total


EmbedderFactory = Callable[[], SentenceTransformer]
