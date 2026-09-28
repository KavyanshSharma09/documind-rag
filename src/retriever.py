from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection
    from sentence_transformers import SentenceTransformer

from src.ingest import embed_texts


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    score: float


class RetrievalMode(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class RetrieverState:
    chunk_ids: list[str]
    corpus: list[str]
    metadatas: list[dict]
    bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def rebuild_bm25_index(collection: Collection) -> RetrieverState:
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
    res: list[RetrievedChunk] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for i, _id in enumerate(ids):
        meta = metadatas[i] or {}
        dst = distances[i] if i < len(distances) else 0.0
        sco = 1.0 / (1.0 + float(dst))
        res.append(
            RetrievedChunk(
                text=documents[i] or "",
                source_file=str(meta.get("source_file", "unknown")),
                page_number=int(meta.get("page_number", 1)),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=sco,
            )
        )
    return res


def retrieve_vector(
    collection: Collection,
    embedder: SentenceTransformer,
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
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
    if not state.bm25 or not state.corpus:
        return []

    scos = state.bm25.get_scores(_tokenize(query))
    rnk = sorted(range(len(scos)), key=lambda i: scos[i], reverse=True)[:top_k]
    res: list[RetrievedChunk] = []
    for i in rnk:
        meta = state.metadatas[i] or {}
        res.append(
            RetrievedChunk(
                text=state.corpus[i],
                source_file=str(meta.get("source_file", "unknown")),
                page_number=int(meta.get("page_number", 1)),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=float(scos[i]),
            )
        )
    return res


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
    top_k: int = 4,
) -> list[RetrievedChunk]:
    fs: dict[tuple[str, int, int], float] = {}
    cm: dict[tuple[str, int, int], RetrievedChunk] = {}

    for rnk in ranked_lists:
        for i, chun in enumerate(rnk, start=1):
            key = (chun.source_file, chun.page_number, chun.chunk_index)
            fs[key] = fs.get(key, 0.0) + 1.0 / (k + i)
            cm[key] = chun

    ordered = sorted(fs.items(), key=lambda x: x[1], reverse=True)
    res: list[RetrievedChunk] = []
    for key, sco in ordered[:top_k]:
        base = cm[key]
        res.append(
            RetrievedChunk(
                text=base.text,
                source_file=base.source_file,
                page_number=base.page_number,
                chunk_index=base.chunk_index,
                score=sco,
            )
        )
    return res


def retrieve(
    mode: RetrievalMode,
    collection: Collection,
    embedder: SentenceTransformer,
    state: RetrieverState,
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    if mode == RetrievalMode.VECTOR:
        return retrieve_vector(collection, embedder, query, top_k)
    vector_hits = retrieve_vector(collection, embedder, query, top_k=top_k * 2)
    bm25_hits = retrieve_bm25(state, query, top_k=top_k * 2)
    return reciprocal_rank_fusion([vector_hits, bm25_hits], top_k=top_k)
