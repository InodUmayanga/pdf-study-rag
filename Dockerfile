FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py config.py ingest.py prompts.py rag.py utils.py ./

EXPOSE 8501

# PDFs and the vector DB are mounted at runtime:
#   docker run -p 8501:8501 --env-file .env \
#     -v ./pdfs:/app/pdfs -v ./chroma_db:/app/chroma_db pdf-study-rag
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
