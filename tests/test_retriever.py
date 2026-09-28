"""Retriever unit tests (no network or API calls)."""

from __future__ import annotations

from src.retriever import (
    RetrievedChunk,
    RetrieverState,
    reciprocal_rank_fusion,
    retrieve_bm25,
)


def _chunk(text: str, page: int, idx: int) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source_file="sample.md",
        page_number=page,
        chunk_index=idx,
        score=1.0,
    )


def test_reciprocal_rank_fusion_prefers_consensus() -> None:
    a = _chunk("alpha document", 1, 0)
    b = _chunk("beta document", 1, 1)
    list_a = [a, b]
    list_b = [b, a]
    fused = reciprocal_rank_fusion([list_a, list_b], top_k=2)
    assert len(fused) == 2
    texts = {c.chunk_index for c in fused}
    assert texts == {0, 1}


def test_bm25_finds_keyword_match() -> None:
    corpus = [
        "DocuMind supports hybrid retrieval with BM25 and vectors.",
        "Unrelated content about cooking pasta.",
    ]
    metadatas = [
        {"source_file": "a.md", "page_number": 1, "chunk_index": 0},
        {"source_file": "a.md", "page_number": 2, "chunk_index": 1},
    ]
    from rank_bm25 import BM25Okapi

    tokenized = [doc.lower().split() for doc in corpus]
    state = RetrieverState(
        chunk_ids=["1", "2"],
        corpus=corpus,
        metadatas=metadatas,
        bm25=BM25Okapi(tokenized),
    )
    hits = retrieve_bm25(state, "BM25 hybrid retrieval", top_k=1)
    assert hits
    assert hits[0].chunk_index == 0


def test_rrf_single_list_preserves_order() -> None:
    chunks = [_chunk("one", 1, 0), _chunk("two", 1, 1)]
    out = reciprocal_rank_fusion([chunks], top_k=2)
    assert [c.chunk_index for c in out] == [0, 1]
