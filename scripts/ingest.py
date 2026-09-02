"""Builds the Chroma vector index from the PDFs in policies_docs.

Run this once before starting the Streamlit app, and again any time
the documents in policies_docs change:

    python scripts/ingest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import chunk_documents
from rag.config import load_config
from rag.document_loader import load_all_documents
from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore


def run_ingestion() -> None:
    config = load_config()

    print(f"Loading PDFs from {config.documents.source_directory} ...")
    pages = load_all_documents(config.documents.source_directory)
    print(f"Loaded {len(pages)} pages.")

    print("Chunking pages ...")
    chunks = chunk_documents(
        pages,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        separators=config.chunking.separators,
    )
    print(f"Created {len(chunks)} chunks.")

    print(f"Loading embedding model {config.embedding.model_name} ...")
    embedding_generator = EmbeddingGenerator(
        config.embedding.model_name, batch_size=config.embedding.batch_size
    )

    print("Embedding chunks ...")
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_generator.embed_texts(texts)

    ids = [
        f"{chunk.source_file}::page-{chunk.page_number}::chunk-{chunk.chunk_index}"
        for chunk in chunks
    ]
    metadatas = [
        {
            "source_file": chunk.source_file,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]

    print(f"Writing to Chroma collection '{config.chroma.collection_name}' ...")
    vector_store = VectorStore(
        persist_directory=config.chroma.persist_directory,
        collection_name=config.chroma.collection_name,
    )
    vector_store.reset()
    vector_store.add_chunks(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

    print(f"Done. Collection now has {vector_store.count()} chunks.")


if __name__ == "__main__":
    run_ingestion()
