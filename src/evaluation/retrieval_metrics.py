"""Retrieval evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievalMetricSummary:
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    evaluated_queries: int


def _dcg(relevance: list[int]) -> float:
    scores = np.asarray(relevance, dtype=float)
    if scores.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, scores.size + 2))
    return float(np.sum((2 ** scores - 1) / discounts))


def compute_retrieval_metrics(
    ranked_results: list[list[int]],
    *,
    k: int = 5,
) -> RetrievalMetricSummary:
    """Compute retrieval quality metrics from ranked relevance labels."""
    if not ranked_results:
        return RetrievalMetricSummary(0.0, 0.0, 0.0, 0.0, 0)

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []

    for relevance in ranked_results:
        top_k = relevance[:k]
        positives_total = int(sum(relevance))
        positives_at_k = int(sum(top_k))

        precision_scores.append(positives_at_k / max(1, k))
        recall_scores.append(positives_at_k / max(1, positives_total))

        reciprocal_rank = 0.0
        for index, value in enumerate(relevance, start=1):
            if value:
                reciprocal_rank = 1.0 / index
                break
        mrr_scores.append(reciprocal_rank)

        ideal = sorted(relevance, reverse=True)[:k]
        ndcg_scores.append(_dcg(top_k) / max(_dcg(ideal), 1e-9))

    return RetrievalMetricSummary(
        precision_at_k=float(np.mean(precision_scores)),
        recall_at_k=float(np.mean(recall_scores)),
        mean_reciprocal_rank=float(np.mean(mrr_scores)),
        ndcg_at_k=float(np.mean(ndcg_scores)),
        evaluated_queries=len(ranked_results),
    )
