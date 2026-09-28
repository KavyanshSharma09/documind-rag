"""Generate the deterministic sample PDF used by the evaluation harness."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


PAGES = [
    (
        "DocuMind Handbook: Retrieval-Augmented Generation",
        [
            "DocuMind is a small reference system for asking questions about documents.",
            "Retrieval-augmented generation, often called RAG, retrieves relevant text",
            "before asking a language model to write an answer. This handbook is a",
            "self-contained sample document for reproducible evaluation experiments.",
        ],
    ),
    (
        "Ingestion and chunking",
        [
            "The ingestion pipeline extracts PDF text page by page and treats a text",
            "or Markdown file as one logical page. Text is split into overlapping",
            "character chunks so that a fact near a boundary remains discoverable.",
            "The default chunk size is 800 characters and the default overlap is 100.",
        ],
    ),
    (
        "Embeddings and storage",
        [
            "DocuMind embeds every chunk with all-MiniLM-L6-v2 from sentence-transformers.",
            "The vectors and their metadata are stored in a persistent local ChromaDB",
            "collection. Metadata records the source filename, one-indexed page number,",
            "and chunk index so that answers can display useful citations.",
        ],
    ),
    (
        "Retrieval modes",
        [
            "Vector retrieval compares the question embedding with stored chunk vectors.",
            "Hybrid retrieval combines vector results with lexical BM25 results.",
            "Reciprocal Rank Fusion, or RRF, gives each result 1 divided by k plus rank",
            "and rewards chunks that appear near the top of both ranked lists.",
        ],
    ),
    (
        "Grounded answers",
        [
            "The answer prompt instructs the model to use only the retrieved context.",
            "A source citation uses the format [filename, p.N]. When the context does",
            "not contain an answer, the required response is: I couldn't find this in",
            "the documents. Showing the source chunk makes each citation inspectable.",
        ],
    ),
    (
        "Evaluation and operations",
        [
            "The evaluation harness measures retrieval hit rate at k and mean reciprocal",
            "rank, abbreviated MRR. It can also ask a configured language model to judge",
            "answer quality on a one-to-five scale. Results are saved as CSV for README",
            "tables. The local index can be cleared from the application sidebar.",
        ],
    ),
]


def generate(output_path: Path) -> None:
    """Write the sample handbook PDF to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    for title, lines in PAGES:
        pdf.setTitle("DocuMind Handbook")
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(54, height - 64, title)
        pdf.setFont("Helvetica", 11)
        for line_number, line in enumerate(lines):
            pdf.drawString(54, height - 100 - line_number * 24, line)
        pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    generate(Path(__file__).parents[1] / "data" / "sample_docs" / "documind_handbook.pdf")