"""Run retrieval and optional LLM-judge evaluation across DocuMind configs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import chromadb

from src.chain import generate_answer
from src.ingest import build_embedder, chunk_pages, get_or_create_collection, index_chunks, load_pdf
from src.llm import LLMError, get_llm_provider
from src.retriever import RetrievalMode, RetrieverState, rebuild_bm25_index, retrieve


ROOT = Path(__file__).parents[1]
QUESTIONS_PATH = ROOT / "eval" / "questions.json"
SAMPLE_PDF = ROOT / "data" / "sample_docs" / "documind_handbook.pdf"
RESULTS_PATH = ROOT / "eval" / "results.csv"


def load_questions() -> list[dict[str, Any]]:
    """Load and minimally validate the question dataset."""
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or len(questions) != 20:
        raise ValueError("eval/questions.json must contain exactly 20 questions.")
    return questions


def reciprocal_rank(chunks: list[Any], source_page: int) -> float:
    """Return the reciprocal rank of the target page, or zero if absent."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.page_number == source_page:
            return 1.0 / rank
    return 0.0


def judge_prompt(question: str, answer: str, ground_truth: str) -> str:
    """Build a strict one-to-five scoring prompt."""
    return (
        "Score the answer from 1 to 5 for factual agreement with the reference. "
        "Return only one integer.\n"
        f"Question: {question}\nReference: {ground_truth}\nAnswer: {answer}"
    )


def judge_score(llm: Any, question: str, answer: str, ground_truth: str) -> float | None:
    """Ask the configured provider for a score, returning None on malformed output."""
    try:
        raw = llm.complete(judge_prompt(question, answer, ground_truth)).strip()
        score = float(raw.split()[0])
    except (ValueError, IndexError, LLMError):
        return None
    return score if 1 <= score <= 5 else None


def evaluate_configuration(
    pages: list[Any],
    questions: list[dict[str, Any]],
    embedder: Any,
    chunk_size: int,
    mode: RetrievalMode,
    top_k: int,
    llm: Any | None,
) -> dict[str, Any]:
    """Build one temporary index and score every question against it."""
    client = chromadb.EphemeralClient()
    collection = get_or_create_collection(client)
    chunks = chunk_pages(pages, chunk_size, overlap=min(100, chunk_size - 1))
    index_chunks(collection, chunks, embedder)
    state: RetrieverState = rebuild_bm25_index(collection)
    hits = []
    ranks = []
    judge_scores = []
    for item in questions:
        retrieved = retrieve(mode, collection, embedder, state, item["question"], top_k)
        rank = reciprocal_rank(retrieved, int(item["source_page"]))
        hits.append(bool(rank))
        ranks.append(rank)
        if llm is not None:
            answer = generate_answer(llm, item["question"], retrieved)
            score = judge_score(llm, item["question"], answer.text, item["ground_truth"])
            if score is not None:
                judge_scores.append(score)
    return {
        "mode": mode.value,
        "chunk_size": chunk_size,
        "top_k": top_k,
        "hit_rate_at_k": sum(hits) / len(hits),
        "mrr": sum(ranks) / len(ranks),
        "answer_quality": sum(judge_scores) / len(judge_scores) if judge_scores else "",
    }


def write_results(rows: list[dict[str, Any]]) -> None:
    """Write CSV results and print a pasteable Markdown table."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["mode", "chunk_size", "top_k", "hit_rate_at_k", "mrr", "answer_quality"]
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("| mode | chunk size | hit rate@k | MRR | answer quality |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['mode']} | {row['chunk_size']} | {row['hit_rate_at_k']:.3f} | "
            f"{row['mrr']:.3f} | {row['answer_quality'] or 'not run'} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="store_true", help="Enable configured LLM-as-judge calls.")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    if not SAMPLE_PDF.exists():
        raise SystemExit("Sample PDF missing; run scripts/generate_sample_pdf.py first.")
    questions = load_questions()
    pages = load_pdf(SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name)
    embedder = build_embedder()
    llm = None
    if args.judge:
        try:
            llm = get_llm_provider()
        except LLMError as exc:
            print(f"Judge disabled: {exc}")
    rows = [
        evaluate_configuration(pages, questions, embedder, chunk_size, mode, args.top_k, llm)
        for mode in (RetrievalMode.VECTOR, RetrievalMode.HYBRID)
        for chunk_size in (400, 800, 1200)
    ]
    write_results(rows)


if __name__ == "__main__":
    main()