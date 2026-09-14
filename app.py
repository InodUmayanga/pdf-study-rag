"""Streamlit RAG app — ask questions about your PDFs with citations."""

import os
import warnings

warnings.filterwarnings("ignore")

import chromadb
import streamlit as st
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import COLLECTION_NAME, DB_DIR, EMBED_MODEL, LLM_MODEL


@st.cache_resource
def init_index():
    """Load the persisted ChromaDB index."""
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        st.error(
            "GROQ_API_KEY not found. Please set it in your .env file."
        )
        st.stop()

    llm = Groq(api_key=api_key, model=LLM_MODEL)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 64
    Settings.context_window = 32768
    Settings.num_output = 2048

    db = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index


def format_sources(source_nodes):
    """Format source nodes into readable citations."""
    if not source_nodes:
        return ""
    lines = ["\n\n---\n**📚 Sources:**"]
    for i, node in enumerate(source_nodes, 1):
        meta = node.node.metadata or {}
        file_name = meta.get("file_name", "Unknown")
        page = meta.get("page_label", "?")
        score = node.score if node.score is not None else 0.0
        lines.append(
            f"{i}. **{file_name}** — Page {page} (relevance: {score:.2f})"
        )
    return "\n".join(lines)


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

index = init_index()
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="simple_summarize",
)

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
            sources = format_sources(response.source_nodes)
            full_response = str(response) + sources

        st.markdown(full_response)

        # Show source excerpts in expander
        if response.source_nodes:
            with st.expander("View source excerpts"):
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

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
