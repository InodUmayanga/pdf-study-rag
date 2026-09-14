# PDF Study RAG

A local RAG (retrieval-augmented generation) app that lets you chat with your
PDF study materials and get answers with page-level citations.

Built with **LlamaIndex** + **ChromaDB** + **Streamlit**. Answers are generated
by **Groq** (llama-3.3-70b-versatile) and embeddings run locally with
**BAAI/bge-small-en-v1.5** — so the only API key you need is a free Groq key.

Because the source PDFs are image-based scans/slides, ingestion renders each
page with **PyMuPDF** and extracts text with **RapidOCR** (ONNX, fully local —
no Tesseract install needed).

## Setup

```bash
# 1. Create a virtual environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
cp .env.example .env             # then edit .env and paste your key
# Get a free key at https://console.groq.com
```

## Usage

```bash
# 1. Put your PDFs into the pdfs/ folder

# 2. Ingest them into the vector database (OCR — takes a few minutes)
python ingest.py

# 3. Run the app
streamlit run app.py
```

Then open http://localhost:8501 and ask questions about your documents.
Each answer includes citations (file name, page, relevance score) and an
expandable view of the source excerpts.

## How it works

| Step | Tool |
|---|---|
| PDF → images | PyMuPDF (~144 DPI render) |
| Images → text | RapidOCR (ONNX runtime, local) |
| Text → vectors | HuggingFace `bge-small-en-v1.5` (local) |
| Vector store | ChromaDB (`./chroma_db`) |
| Q&A LLM | Groq `llama-3.3-70b-versatile` |
| UI | Streamlit chat |

Re-running `python ingest.py` rebuilds the collection from scratch.

## Project layout

```
├── app.py             # Streamlit chat app with citations
├── ingest.py          # OCR + embedding pipeline → ChromaDB
├── requirements.txt
├── .env.example       # copy to .env and add GROQ_API_KEY
└── pdfs/              # drop your PDFs here (not committed to git)
```

## Notes

- `pdfs/`, `chroma_db/`, `.env`, and `.venv/` are gitignored — your documents
  and API key never leave your machine.
- OCR quality depends on the source images; if results look poor, bump
  `RENDER_SCALE` in `ingest.py` from `2` to `3`.
