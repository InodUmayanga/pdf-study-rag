"""Query-side wiring: open the persisted index and build a grounded query
engine.

Separate from ``app.py`` so the evaluation harness can reuse the exact same
retrieval and prompting code without Streamlit.
"""

import chromadb
from llama_index.core import (
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

from prompts import GROUNDED_QA_PROMPT, GROUNDED_REFINE_PROMPT

GROUNDED_QA_TEMPLATE = PromptTemplate(GROUNDED_QA_PROMPT)
GROUNDED_REFINE_TEMPLATE = PromptTemplate(GROUNDED_REFINE_PROMPT)


def make_llm(api_key, model, context_window=32768, num_output=2048):
    """Groq chat model, registered as the llama-index default LLM."""
    llm = Groq(api_key=api_key, model=model)
    Settings.llm = llm
    Settings.context_window = context_window
    Settings.num_output = num_output
    return llm


def resolve_embed_model(embed_model):
    """Accept a model name or an embedding object; return the object."""
    if isinstance(embed_model, str):
        return HuggingFaceEmbedding(model_name=embed_model)
    return embed_model


def load_index(db_dir, collection_name, embed_model):
    """Open an existing ChromaDB collection as a llama-index index.

    ``embed_model`` may be a HuggingFace model name or an embedding object.
    It must match the model used at ingestion time.
    """
    embed_model = resolve_embed_model(embed_model)
    Settings.embed_model = embed_model

    db = chromadb.PersistentClient(path=db_dir)
    collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )


def build_retriever(index, top_k=3):
    """Plain top-k retriever (no LLM) — used by the evaluation harness."""
    return index.as_retriever(similarity_top_k=top_k)


def build_query_engine(index, top_k=3, llm=None):
    """Query engine that answers only from retrieved passages and cites
    them as [file_name p.page]; returns the fixed refusal sentence when the
    passages do not contain the answer (see ``prompts.NOT_COVERED``).
    """
    kwargs = {
        "similarity_top_k": top_k,
        "response_mode": "compact",
        "text_qa_template": GROUNDED_QA_TEMPLATE,
        "refine_template": GROUNDED_REFINE_TEMPLATE,
    }
    if llm is not None:
        kwargs["llm"] = llm
    return index.as_query_engine(**kwargs)
