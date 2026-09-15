"""Streamlit RAG app — ask questions about your PDFs with citations."""

import os
import warnings

warnings.filterwarnings("ignore")

import streamlit as st

from config import COLLECTION_NAME, DB_DIR, EMBED_MODEL, LLM_MODEL, TOP_K
from prompts import is_not_covered, normalize_citations
from rag import build_query_engine, load_index, make_llm
from utils import format_sources


@st.cache_resource
def init_query_engine():
    """Open the persisted index and build the grounded query engine."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        st.error(
            "GROQ_API_KEY not found. Please set it in your .env file."
        )
        st.stop()

    llm = make_llm(api_key, LLM_MODEL)
    index = load_index(DB_DIR, COLLECTION_NAME, EMBED_MODEL)
    return build_query_engine(index, top_k=TOP_K, llm=llm)


# --- Streamlit UI ---
st.set_page_config(
    page_title="PDF Study RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF Study RAG")
st.caption("Ask questions about your PDFs — answers with citations.")

# Check DB exists
if not os.path.isdir(DB_DIR):
    st.warning(
        "Vector database not found. Run ingestion first:\n\n"
        "`python ingest.py`"
    )
    st.stop()

query_engine = init_query_engine()

# --- Chat session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs..."):
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching your PDFs..."):
            try:
                response = query_engine.query(prompt)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                st.stop()
            answer = normalize_citations(str(response))
            not_covered = is_not_covered(answer)
            if not_covered:
                # Nothing relevant was found; don't present the nearest
                # passages as if they supported an answer.
                full_response = answer
            else:
                full_response = answer + format_sources(
                    response.source_nodes
                )

        st.markdown(full_response)

        # Show source excerpts in expander
        if response.source_nodes:
            label = (
                "Closest passages (not used — no answer found)"
                if not_covered
                else "View source excerpts"
            )
            with st.expander(label):
                for i, node in enumerate(
                    response.source_nodes, 1
                ):
                    meta = node.node.metadata or {}
                    file_name = meta.get("file_name", "Unknown")
                    page = meta.get("page_label", "?")
                    text = node.node.text[:500] + "..."
                    st.markdown(
                        f"**Source {i}: {file_name} — Page {page}**"
                    )
                    st.text(text)
                    st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses LlamaIndex + ChromaDB + Groq "
        "to answer questions from your PDFs."
    )
    st.write(f"**Embedding model:** `{EMBED_MODEL}`")
    st.write(f"**LLM:** Groq (`{LLM_MODEL}`)")
    st.write(f"**Vector DB:** `{DB_DIR}`")
    st.write(f"**Passages per question:** {TOP_K}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
