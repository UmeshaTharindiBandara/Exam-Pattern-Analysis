"""Cross-encoder reranking for retrieved exam content."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class RerankedResult:
    """Result after reranking."""

    text: str
    score: float
    base_score: float
    metadata: dict


class QuestionReranker:
    """Rerank candidate passages with a cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ValueError(
                    "sentence-transformers is required for reranking."
                ) from exc

            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """Return the top ranked candidates for a query."""
        if candidates.empty:
            return candidates

        working = candidates.copy().reset_index(drop=True)
        pairs = [(query, str(text)) for text in working["text"].fillna("").astype(str).tolist()]
        scores = self.model.predict(pairs)
        working["rerank_score"] = scores
        if "score" not in working.columns:
            working["score"] = 0.0
        ordered = working.sort_values(["rerank_score", "score"], ascending=False)
        return ordered.head(top_k).reset_index(drop=True)
