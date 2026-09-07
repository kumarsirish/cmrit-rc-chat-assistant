"""Step 3a: Sentence embeddings.

Wraps a local sentence-transformers model so the rest of the
application never has to know which embedding model is in use. The
model name is read from configuration, so it can be swapped for a
different sentence-transformers model without changing any other file.
"""

from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

# sentence-transformers pulls in transformers, which logs harmless
# lazy-import alias warnings (e.g. "Accessing `...Fast` ... Returning
# `...` instead") on import. They're internal library noise, not
# something this app can act on, so quiet them here where
# transformers first gets loaded.
hf_logging.set_verbosity_error()


class EmbeddingGenerator:
    """Turns text into vector embeddings using a local model."""

    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list) -> list:
        """Embeds a list of strings in batches and returns a list of float vectors.

        Batching (instead of one encode() call per text) lets the
        model process several chunks per forward pass, which is
        essentially free CPU throughput to reclaim during ingestion.
        """
        embeddings = self._model.encode(
            texts, batch_size=self._batch_size, show_progress_bar=False
        )
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> list:
        """Embeds a single query string and returns one float vector."""
        return self.embed_texts([text])[0]
