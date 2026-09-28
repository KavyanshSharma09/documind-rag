"""Vector, BM25, and hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi
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
    """In-memory BM25 index aligned with Chroma documents."""

    chunk_ids: list[str]
    corpus: list[str]
    metadatas: list[dict]
    bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def rebuild_bm25_index(collection: Collection) -> RetrieverState:
    """Load all documents from Chroma and build a BM25 index."""
    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if not documents:
        return RetrieverState(chunk_ids=[], corpus=[], metadatas=[], bm25=None)

    tokenized = [_tokenize(doc) for doc in documents]
    return RetrieverState(
        chunk_ids=list(ids),
        corpus=list(documents),
        metadatas=list(metadatas),
        bm25=BM25Okapi(tokenized),
    )


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


def retrieve_bm25(
    state: RetrieverState,
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Lexical BM25 retrieval over the in-memory corpus."""
    if not state.bm25 or not state.corpus:
        return []

    scores = state.bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    chunks: list[RetrievedChunk] = []
    for rank_idx in ranked:
        if scores[rank_idx] <= 0:
            continue
        meta = state.metadatas[rank_idx] or {}
        chunks.append(
            RetrievedChunk(
                text=state.corpus[rank_idx],
                source_file=str(meta.get("source_file", "unknown")),
                page_number=int(meta.get("page_number", 1)),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=float(scores[rank_idx]),
            )
        )
    return chunks


def retrieve(
    mode: RetrievalMode,
    collection: Collection,
    embedder: SentenceTransformer,
    state: RetrieverState,
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Dispatch retrieval by mode."""
    if mode == RetrievalMode.HYBRID:
        return retrieve_bm25(state, query, top_k)
    return retrieve_vector(collection, embedder, query, top_k)
