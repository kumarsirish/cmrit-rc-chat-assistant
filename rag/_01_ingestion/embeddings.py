"""Step 3a: Sentence embeddings.

Wraps LangChain's HuggingFaceEmbeddings (itself a thin layer over a
local sentence-transformers model) so the rest of the application
never has to know which embedding model is in use. The model name is
read from configuration, so it can be swapped for a different
sentence-transformers model without changing any other file.
"""

import warnings

# langchain_core falls back to the pydantic.v1 compat shim for some
# internal type checks; that shim isn't validated on Python 3.14+ and
# warns about it on import. It's noise this app can't act on, so quiet
# it here before langchain_core (imported transitively below) fires it.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
    category=UserWarning,
)

from langchain_huggingface import HuggingFaceEmbeddings
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
        self.batch_size = batch_size
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"batch_size": batch_size},
            show_progress=False,
        )

    def embed_texts(self, texts: list, progress_callback=None) -> list:
        """Embeds a list of strings in batches and returns a list of float vectors.

        Batching (instead of one encode() call per text) lets the
        model process several chunks per forward pass, which is
        essentially free CPU throughput to reclaim during ingestion.

        progress_callback(completed_count, total_count), if given, is
        called after each batch — this is what lets a slow, single
        long call (e.g. embedding a whole document set during
        ingestion) report real, incremental progress instead of
        leaving a caller's progress bar frozen until everything
        finishes. Without it, every text is still batched internally
        by the model (via encode_kwargs's batch_size) in one call, so
        batching here instead only adds the ability to observe
        progress between batches — not a different embedding result.
        """
        if progress_callback is None:
            return self._model.embed_documents(texts)

        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(self._model.embed_documents(batch))
            progress_callback(len(embeddings), len(texts))
        return embeddings

    def embed_query(self, text: str) -> list:
        """Embeds a single query string and returns one float vector."""
        return self._model.embed_query(text)
