from __future__ import annotations

from dataclasses import dataclass

from src.config import NOT_FOUND_PHRASE, PROMPT_TEMPLATE, SYSTEM_INSTRUCTION
from src.llm import BaseLLMProvider
from src.retriever import RetrievedChunk


@dataclass(frozen=True)
class Citation:
    source_file: str
    page_number: int
    excerpt: str


@dataclass(frozen=True)
class RAGAnswer:
    text: str
    citations: list[Citation]
    chunks: list[RetrievedChunk]


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no context retrieved)"
    res: list[str] = []
    for i, chun in enumerate(chunks, start=1):
        res.append(
            f"[{i}] Source: [{chun.source_file}, p.{chun.page_number}]\n{chun.text}"
        )
    return "\n\n".join(res)


def chunks_to_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            source_file=c.source_file,
            page_number=c.page_number,
            excerpt=c.text,
        )
        for c in chunks
    ]


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return PROMPT_TEMPLATE.format(
        system_instruction=SYSTEM_INSTRUCTION,
        context=format_context(chunks),
        question=question,
    )


def generate_answer(
    llm: BaseLLMProvider,
    question: str,
    chunks: list[RetrievedChunk],
) -> RAGAnswer:
    if not chunks:
        return RAGAnswer(
            text=NOT_FOUND_PHRASE,
            citations=[],
            chunks=[],
        )

    prompt = build_prompt(question, chunks)
    raw = llm.complete(prompt)
    normalized = raw.strip()
    if not normalized:
        normalized = NOT_FOUND_PHRASE

    return RAGAnswer(
        text=normalized,
        citations=chunks_to_citations(chunks),
        chunks=chunks,
    )
