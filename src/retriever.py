"""Vector, BM25, and hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from src.ingest import embed_texts


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from retrieval with score and citation metadata."""

    text: str
    source_file: str
    page_number: int
    chunk_index: int
    score: float


class RetrievalMode(str, Enum):
    """Supported retrieval strategies."""

    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class RetrieverState:
    """Placeholder for BM25 state (extended in hybrid retrieval)."""

    chunk_ids: list[str]
    corpus: list[str]
    metadatas: list[dict]


def _rows_from_chroma_result(result: dict) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for idx, _doc_id in enumerate(ids):
        meta = metadatas[idx] or {}
        dist = distances[idx] if idx < len(distances) else 0.0
        score = 1.0 / (1.0 + float(dist))
        chunks.append(
            RetrievedChunk(
                text=documents[idx] or "",
                source_file=str(meta.get("source_file", "unknown")),
                page_number=int(meta.get("page_number", 1)),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=score,
            )
        )
    return chunks


def retrieve_vector(
    collection: Collection,
    embedder: SentenceTransformer,
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Cosine similarity search in Chroma."""
    if collection.count() == 0:
        return []

    query_embedding = embed_texts(embedder, [query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return _rows_from_chroma_result(result)


def retrieve(
    mode: RetrievalMode,
    collection: Collection,
    embedder: SentenceTransformer,
    state: RetrieverState,
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Dispatch retrieval by mode (hybrid added in a later commit)."""
    _ = state
    if mode == RetrievalMode.HYBRID:
        return retrieve_vector(collection, embedder, query, top_k)
    return retrieve_vector(collection, embedder, query, top_k)
