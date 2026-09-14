"""Ingest all PDFs in ./pdfs/ into a local ChromaDB vector store.

The PDFs are image-based (no text layer), so each page is rendered with
PyMuPDF and OCR'd with RapidOCR (ONNX, runs locally — no Tesseract needed).
Each page becomes one Document with file_name / page_label metadata.
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
from llama_index.llms.groq import Groq
from rapidocr_onnxruntime import RapidOCR

from config import (
    COLLECTION_NAME,
    DB_DIR,
    EMBED_MODEL,
    LLM_MODEL,
    PDF_DIR,
    RENDER_SCALE,
)


def extract_documents(pdf_dir):
    """OCR every page of every PDF into a llama-index Document."""
    ocr = RapidOCR()
    documents = []
    pdf_files = sorted(
        f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")
    )

    for file_name in pdf_files:
        path = os.path.join(pdf_dir, file_name)
        pdf = pymupdf.open(path)
        print(f"  {file_name}: {len(pdf)} page(s)")
        for page_num, page in enumerate(pdf, 1):
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE),
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


def main():
    if not os.path.isdir(PDF_DIR):
        print(f"ERROR: '{PDF_DIR}' directory not found.")
        sys.exit(1)

    pdf_files = [
        f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")
    ]
    print(f"Found {len(pdf_files)} PDF(s) in '{PDF_DIR}/'")

    # --- Embedding model (local, free) ---
    print(f"Loading embedding model: {EMBED_MODEL} ...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # --- LLM (Groq) ---
    api_key = os.getenv("GROQ_API_KEY", "")
    llm = None
    if api_key:
        llm = Groq(api_key=api_key, model=LLM_MODEL)
        print("Groq LLM configured.")
    else:
        print(
            "WARNING: GROQ_API_KEY not set. "
            "Ingestion will proceed (embeddings only); "
            "LLM will be configured at query time in app.py."
        )

    # --- Global settings ---
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 128

    # --- ChromaDB client (fresh rebuild each run) ---
    print(f"Initializing ChromaDB at '{DB_DIR}' ...")
    db = chromadb.PersistentClient(path=DB_DIR)
    existing = {c.name for c in db.list_collections()}
    if COLLECTION_NAME in existing:
        db.delete_collection(COLLECTION_NAME)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # --- OCR PDFs ---
    print("Extracting text from PDFs via OCR (this can take a while) ...")
    documents = extract_documents(PDF_DIR)
    print(f"Extracted text from {len(documents)} page(s).")

    # --- Build index ---
    print("Generating embeddings and storing in ChromaDB ...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"\nDone! {len(documents)} pages ingested into '{DB_DIR}/'.")
    print("You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
