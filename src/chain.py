"""RAG prompt assembly, grounded generation, and citation formatting."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import NOT_FOUND_PHRASE, PROMPT_TEMPLATE, SYSTEM_INSTRUCTION
from src.llm import BaseLLMProvider
from src.retriever import RetrievedChunk


@dataclass(frozen=True)
class Citation:
    """Structured citation for UI display."""

    source_file: str
    page_number: int
    excerpt: str


@dataclass(frozen=True)
class RAGAnswer:
    """Model answer plus retrieval citations."""

    text: str
    citations: list[Citation]
    chunks: list[RetrievedChunk]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Build numbered context blocks for the prompt."""
    if not chunks:
        return "(no context retrieved)"
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] Source: [{chunk.source_file}, p.{chunk.page_number}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def chunks_to_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Map retrieved chunks to citation objects."""
    return [
        Citation(
            source_file=c.source_file,
            page_number=c.page_number,
            excerpt=c.text,
        )
        for c in chunks
    ]


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Fill the grounded prompt template."""
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
    """Generate a grounded answer with citations from retrieved chunks."""
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
