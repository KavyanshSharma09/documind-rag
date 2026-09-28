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
    text: str
    page_number: int
    source_file: str


def load_pdf(file_bytes: bytes, filename: str) -> list[PageRecord]:
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
    text = content.strip()
    if not text:
        raise ValueError(f"File '{filename}' is empty.")
    return [
        PageRecord(text=text, page_number=1, source_file=filename),
    ]


def load_upload(filename: str, file_bytes: bytes) -> list[PageRecord]:
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
    text: str
    source_file: str
    page_number: int
    chunk_index: int


def chunk_pages(
    pages: list[PageRecord],
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size.")

    res: list[TextChunk] = []
    gi = 0

    for pg in pages:
        text = pg.text
        if not text:
            continue
        x = 0
        while x < len(text):
            y = min(x + chunk_size, len(text))
            part = text[x:y].strip()
            if part:
                res.append(
                    TextChunk(
                        text=part,
                        source_file=pg.source_file,
                        page_number=pg.page_number,
                        chunk_index=gi,
                    )
                )
                gi += 1
            if y >= len(text):
                break
            x = y - overlap

    return res


def get_chroma_client() -> chromadb.PersistentClient:
    import chromadb

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def get_or_create_collection(client: chromadb.PersistentClient) -> Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(
    embedder: SentenceTransformer,
    texts: list[str],
) -> list[list[float]]:
    vectors = embedder.encode(texts, show_progress_bar=False)
    return vectors.tolist()


def index_chunks(
    collection: Collection,
    chunks: list[TextChunk],
    embedder: SentenceTransformer,
) -> int:
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
    total = 0
    for filename, data in files:
        pages = load_upload(filename, data)
        chunks = chunk_pages(pages, chunk_size, overlap)
        if not chunks:
            raise ValueError(f"No text extracted from '{filename}'.")
        total += index_chunks(collection, chunks, embedder)
    return total


EmbedderFactory = Callable[[], "SentenceTransformer"]
