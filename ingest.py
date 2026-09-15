"""Ingest all PDFs in ./pdfs/ into a local ChromaDB vector store.

The PDFs are image-based (no text layer), so each page is rendered with
PyMuPDF and OCR'd with RapidOCR (ONNX, runs locally — no Tesseract needed).
Each page becomes one Document with file_name / page_label metadata.

``build_index`` is the reusable entry point; the evaluation harness in
``evals/`` calls it with a temporary database directory so the eval and
the CLI exercise exactly the same pipeline.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import chromadb
import numpy as np
import pymupdf
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from rapidocr_onnxruntime import RapidOCR

from config import (
    COLLECTION_NAME,
    DB_DIR,
    EMBED_MODEL,
    PDF_DIR,
    RENDER_SCALE,
)

# Chunking used at ingestion time. Retrieval quality is measured against
# these values (see evals/), so change them deliberately.
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128


def list_pdfs(pdf_dir):
    """Sorted PDF file names in ``pdf_dir`` (deterministic order)."""
    return sorted(
        f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")
    )


def extract_documents(pdf_dir, render_scale=RENDER_SCALE, ocr=None):
    """OCR every page of every PDF into a llama-index Document."""
    ocr = ocr or RapidOCR()
    documents = []

    for file_name in list_pdfs(pdf_dir):
        path = os.path.join(pdf_dir, file_name)
        pdf = pymupdf.open(path)
        print(f"  {file_name}: {len(pdf)} page(s)")
        for page_num, page in enumerate(pdf, 1):
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(render_scale, render_scale),
                alpha=False,
            )
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            result, _ = ocr(img)
            text = "\n".join(line[1] for line in (result or [])).strip()
            if not text:
                continue
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": file_name,
                        "page_label": str(page_num),
                    },
                )
            )
        pdf.close()
    return documents


def build_index(
    pdf_dir,
    db_dir,
    collection_name,
    embed_model=EMBED_MODEL,
    render_scale=RENDER_SCALE,
    show_progress=False,
):
    """OCR ``pdf_dir`` and (re)build the ChromaDB collection from scratch.

    Returns ``(index, documents)``. ``embed_model`` may be a model name or
    an embedding object.
    """
    if isinstance(embed_model, str):
        embed_model = HuggingFaceEmbedding(model_name=embed_model)

    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    db = chromadb.PersistentClient(path=db_dir)
    existing = {
        c if isinstance(c, str) else c.name for c in db.list_collections()
    }
    if collection_name in existing:
        db.delete_collection(collection_name)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    documents = extract_documents(pdf_dir, render_scale=render_scale)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=show_progress,
    )
    return index, documents


def main():
    if not os.path.isdir(PDF_DIR):
        print(f"ERROR: '{PDF_DIR}' directory not found.")
        sys.exit(1)

    pdf_files = list_pdfs(PDF_DIR)
    print(f"Found {len(pdf_files)} PDF(s) in '{PDF_DIR}/'")
    if not pdf_files:
        print("Nothing to ingest — add PDFs to the folder and re-run.")
        sys.exit(1)

    print(f"Loading embedding model: {EMBED_MODEL} ...")
    print(f"Initializing ChromaDB at '{DB_DIR}' ...")
    print("Extracting text from PDFs via OCR (this can take a while) ...")
    _, documents = build_index(
        PDF_DIR, DB_DIR, COLLECTION_NAME, show_progress=True
    )

    print(f"\nDone! {len(documents)} pages ingested into '{DB_DIR}/'.")
    print("The Groq LLM is configured at query time (streamlit run app.py).")


if __name__ == "__main__":
    main()
