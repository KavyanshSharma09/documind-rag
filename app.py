"""DocuMind Streamlit application."""

from __future__ import annotations

import streamlit as st
from sentence_transformers import SentenceTransformer

from src.chain import generate_answer
from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    LLM_PROVIDER,
)
from src.ingest import (
    build_embedder,
    clear_index,
    get_chroma_client,
    get_or_create_collection,
    ingest_uploads,
)
from src.retriever import rebuild_bm25_index
from src.llm import LLMError, get_llm_provider
from src.retriever import RetrievalMode, RetrieverState, retrieve


@st.cache_resource(show_spinner="Loading embedding model…")
def cached_embedder() -> SentenceTransformer:
    return build_embedder()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever_state" not in st.session_state:
        st.session_state.retriever_state = RetrieverState([], [], [])


def main() -> None:
    st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")
    init_session_state()

    st.title("DocuMind")
    st.caption("RAG document Q&A with citations")

    with st.sidebar:
        st.header("Settings")
        st.session_state.chunk_size = st.number_input(
            "Chunk size",
            min_value=200,
            max_value=2000,
            value=int(st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE)),
            step=100,
        )
        st.session_state.chunk_overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=500,
            value=int(st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
            step=50,
        )
        st.session_state.top_k = st.slider(
            "Top-k chunks",
            min_value=1,
            max_value=10,
            value=int(st.session_state.get("top_k", DEFAULT_TOP_K)),
        )
        mode_label = st.selectbox(
            "Retrieval mode",
            options=[RetrievalMode.VECTOR.value, RetrievalMode.HYBRID.value],
            index=0
            if st.session_state.get("retrieval_mode", RetrievalMode.VECTOR)
            == RetrievalMode.VECTOR
            else 1,
        )
        st.session_state.retrieval_mode = RetrievalMode(mode_label)
        st.caption(f"LLM provider: `{LLM_PROVIDER}`")
        if st.button("Clear index", type="secondary"):
            client_tmp = get_chroma_client()
            clear_index(client_tmp)
            st.session_state.retriever_state = RetrieverState([], [], [])
            st.session_state.messages = []
            st.success("Index cleared.")
            st.rerun()

    client = get_chroma_client()
    collection = get_or_create_collection(client)
    embedder = cached_embedder()

    uploaded = st.file_uploader(
        "Upload PDF, TXT, or Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Ingest documents", disabled=not uploaded):
        if not uploaded:
            st.warning("Upload at least one document before ingesting.")
        else:
            files = [(f.name, f.getvalue()) for f in uploaded]
            chunk_size = st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE)
            overlap = st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
            try:
                total = ingest_uploads(
                    files, chunk_size, overlap, embedder, collection
                )
                st.session_state.retriever_state = rebuild_bm25_index(collection)
                st.success(f"Indexed {total} chunks from {len(files)} file(s).")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for cite in message.get("citations", []):
                with st.expander(f"{cite['source_file']} — p.{cite['page_number']}"):
                    st.text(cite["excerpt"])

    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if collection.count() == 0:
                st.warning(
                    "No documents indexed yet. Upload files and click Ingest documents."
                )
                return

            try:
                llm = get_llm_provider()
            except LLMError as exc:
                st.error(str(exc))
                return

            mode = st.session_state.get("retrieval_mode", RetrievalMode.VECTOR)
            top_k = st.session_state.get("top_k", DEFAULT_TOP_K)
            state: RetrieverState = st.session_state.retriever_state
            try:
                chunks = retrieve(mode, collection, embedder, state, prompt, top_k)
                answer = generate_answer(llm, prompt, chunks)
            except LLMError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
                return
            st.markdown(answer.text)
            citation_payload = [
                {
                    "source_file": c.source_file,
                    "page_number": c.page_number,
                    "excerpt": c.excerpt,
                }
                for c in answer.citations
            ]
            for cite in citation_payload:
                with st.expander(f"{cite['source_file']} — p.{cite['page_number']}"):
                    st.text(cite["excerpt"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer.text,
                "citations": citation_payload,
            }
        )


if __name__ == "__main__":
    main()
