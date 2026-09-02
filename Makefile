.PHONY: install download-embedding-model ingest run run-debug test

install:
	pip install -r requirements.txt

# Downloads the sentence-transformers embedding model into the local
# cache (safe to re-run; it's a no-op if already downloaded).
download-embedding-model:
	python scripts/download_embedding_model.py

# Builds/rebuilds the Chroma vector index from policies_docs/.
# Run this once after install, and again whenever the PDFs change.
ingest:
	python scripts/ingest.py

run:
	streamlit run app.py

# Same as run, but with verbose Streamlit logging plus an in-app debug
# panel showing the query, retrieved chunks (source doc/page), and
# their similarity scores.
run-debug:
	RAG_DEBUG=true streamlit run app.py --logger.level=debug

test:
	pytest tests/
