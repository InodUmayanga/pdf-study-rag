"""Shared configuration. Every value can be overridden via .env."""

import os

from dotenv import load_dotenv

load_dotenv()

PDF_DIR = os.getenv("PDF_DIR", "./pdfs")
DB_DIR = os.getenv("DB_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_study_collection")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
# ~144 DPI render for OCR; raise to 3 for small/dense text
RENDER_SCALE = int(os.getenv("RENDER_SCALE", "2"))
# Number of passages retrieved per question and handed to the LLM
TOP_K = int(os.getenv("TOP_K", "3"))
