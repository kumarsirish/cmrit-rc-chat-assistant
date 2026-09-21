"""Downloads (or confirms the local cache of) the embedding and
reranker models.

sentence-transformers/huggingface cache models under
~/.cache/huggingface, so running this is safe and cheap even if the
models are already present:

    python scripts/download_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.common.config import load_config
from rag.ingestion.embeddings import EmbeddingGenerator
from rag.retrieval.reranker import Reranker


def download_models() -> None:
    config = load_config()

    embedding_model_name = config.embedding.model_name
    print(f"Downloading/verifying embedding model '{embedding_model_name}' ...")
    EmbeddingGenerator(embedding_model_name)
    print("Embedding model is ready.")

    if config.retrieval.rerank.enabled:
        reranker_model_name = config.retrieval.rerank.model
        print(f"Downloading/verifying reranker model '{reranker_model_name}' ...")
        Reranker(reranker_model_name)
        print("Reranker model is ready.")
    else:
        print("Reranking is disabled in config.yaml (retrieval.rerank.enabled: false) — skipping.")


if __name__ == "__main__":
    download_models()
