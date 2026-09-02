"""Downloads (or confirms the local cache of) the embedding model.

sentence-transformers caches models under ~/.cache/huggingface, so
running this is safe and cheap even if the model is already present:

    python scripts/download_embedding_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_config
from rag.embeddings import EmbeddingGenerator


def download_embedding_model() -> None:
    config = load_config()
    model_name = config.embedding.model_name
    print(f"Downloading/verifying embedding model '{model_name}' ...")
    EmbeddingGenerator(model_name)
    print("Embedding model is ready.")


if __name__ == "__main__":
    download_embedding_model()
