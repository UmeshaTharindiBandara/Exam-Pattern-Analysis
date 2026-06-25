"""Generate sentence embeddings with caching support."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils import PROCESSED_DIR, setup_logging

logger = setup_logging(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class QuestionEmbedder:
    """Generate and cache semantic embeddings for exam questions."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the embedder.

        Args:
            model_name: Sentence-transformers model identifier.
            cache_dir: Directory for embedding cache files.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or PROCESSED_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _cache_path(self, cache_key: str) -> Path:
        """Build cache file path for a dataset key."""
        safe_key = cache_key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"embeddings_{safe_key}.npy"

    def encode(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for a collection of texts.

        Args:
            texts: Iterable of question strings.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

        Returns:
            Embedding matrix of shape (n_samples, embedding_dim).
        """
        text_list = list(texts)
        if not text_list:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            text_list,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "question_text",
        cache_key: str = "questions",
        force_recompute: bool = False,
    ) -> np.ndarray:
        """Generate or load cached embeddings for a questions dataframe.

        Args:
            df: Questions dataframe.
            text_column: Column containing text to embed.
            cache_key: Cache identifier.
            force_recompute: Recompute even if cache exists.

        Returns:
            Embedding matrix aligned with dataframe rows.
        """
        cache_path = self._cache_path(cache_key)
        meta_path = self.cache_dir / f"embeddings_{cache_key}_meta.csv"

        if cache_path.exists() and meta_path.exists() and not force_recompute:
            cached_meta = pd.read_csv(meta_path)
            if len(cached_meta) == len(df) and cached_meta["question_text"].tolist() == df[
                text_column
            ].tolist():
                logger.info("Loading cached embeddings from %s", cache_path)
                return np.load(cache_path)

        texts = df[text_column].astype(str).tolist()
        embeddings = self.encode(texts, show_progress=False)
        np.save(cache_path, embeddings)
        df[[text_column]].to_csv(meta_path, index=False)
        logger.info("Saved embeddings cache to %s", cache_path)
        return embeddings

    @staticmethod
    def cosine_similarity(
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between one query and a corpus.

        Args:
            query_embedding: Shape (embedding_dim,) or (1, embedding_dim).
            corpus_embeddings: Shape (n_samples, embedding_dim).

        Returns:
            Similarity scores of shape (n_samples,).
        """
        query = query_embedding.reshape(1, -1)
        scores = np.dot(corpus_embeddings, query.T).flatten()
        return scores
