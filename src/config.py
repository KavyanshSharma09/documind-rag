from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 100
DEFAULT_TOP_K: int = 4

EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHROMA_PERSIST_DIR: Path = Path(
    os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
)
COLLECTION_NAME: str = "documind_chunks"

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower().strip()
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

GROQ_MODEL: str = "llama-3.1-8b-instant"
GEMINI_MODEL: str = "gemini-1.5-flash"

SYSTEM_INSTRUCTION: str = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain enough information, respond exactly with: "
    "I couldn't find this in the documents. "
    "When you use information from the context, cite sources inline as [filename, p.N]."
)

PROMPT_TEMPLATE: str = """{system_instruction}

Context:
{context}

Question: {question}

Answer (with citations as [filename, p.N] where applicable):"""

NOT_FOUND_PHRASE: str = "I couldn't find this in the documents."
