"""Topic clustering and trend analysis for exam questions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.utils import setup_logging

logger = setup_logging(__name__)


class TopicClassifier:
    """Cluster question embeddings and label topics using TF-IDF keywords."""

    def __init__(
        self,
        min_clusters: int = 3,
        max_clusters: int = 10,
        random_state: int = 42,
    ) -> None:
        """Initialize clustering configuration.

        Args:
            min_clusters: Minimum clusters to evaluate.
            max_clusters: Maximum clusters to evaluate.
            random_state: Random seed for reproducibility.
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.cluster_model: KMeans | None = None
        self.optimal_k: int = min_clusters

    def find_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Select optimal cluster count using silhouette score.

        Args:
            embeddings: Question embedding matrix.

        Returns:
            Best cluster count.
        """
        n_samples = len(embeddings)
        if n_samples <= 2:
            return 1

        upper = min(self.max_clusters, n_samples - 1)
        lower = min(self.min_clusters, upper)
        if upper < 2:
            return 1

        best_k = lower
        best_score = -1.0

        for k in range(lower, upper + 1):
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k

        self.optimal_k = best_k
        logger.info("Selected %s clusters (silhouette=%.3f)", best_k, best_score)
        return best_k

    def fit(self, embeddings: np.ndarray, n_clusters: int | None = None) -> np.ndarray:
        """Fit KMeans on embeddings and return cluster labels.

        Args:
            embeddings: Question embedding matrix.
            n_clusters: Optional fixed cluster count.

        Returns:
            Cluster label array.
        """
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(embeddings)

        n_clusters = max(1, min(n_clusters, len(embeddings)))
        self.cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.cluster_model.fit_predict(embeddings)
        self.optimal_k = n_clusters
        return labels

    def label_cluster(self, texts: list[str], top_n: int = 3) -> str:
        """Generate a topic label from TF-IDF keywords.

        Args:
            texts: Question texts in the cluster.
            top_n: Number of keywords to include in label.

        Returns:
            Human-readable topic label.
        """
        if not texts:
            return "General"

        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return "General"

        scores = np.asarray(matrix.sum(axis=0)).flatten()
        terms = vectorizer.get_feature_names_out()
        top_indices = scores.argsort()[::-1][:top_n]
        keywords = [terms[i] for i in top_indices if scores[i] > 0]
        if not keywords:
            return "General"
        return " / ".join(keywords).title()

    def build_topic_summary(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        text_column: str = "cleaned_text",
    ) -> pd.DataFrame:
        """Create topic summary with labels and yearly frequency.

        Args:
            df: Enriched questions dataframe.
            labels: Cluster labels aligned with df rows.
            text_column: Text column for keyword extraction.

        Returns:
            Topic summary dataframe.
        """
        working = df.copy()
        working["topic_id"] = labels
        summaries: list[dict[str, Any]] = []

        for topic_id in sorted(working["topic_id"].unique()):
            cluster_df = working[working["topic_id"] == topic_id]
            texts = cluster_df[text_column].fillna(cluster_df["question_text"]).tolist()
            label = self.label_cluster(texts)

            frequency_by_year = (
                cluster_df.groupby("year").size().astype(int).to_dict()
            )
            summaries.append(
                {
                    "topic_id": int(topic_id),
                    "topic_label": label,
                    "question_count": len(cluster_df),
                    "frequency_by_year": frequency_by_year,
                    "sample_questions": cluster_df["question_text"].head(3).tolist(),
                }
            )

        summary_df = pd.DataFrame(summaries)
        return summary_df.sort_values("question_count", ascending=False).reset_index(drop=True)

    def annotate_questions(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        n_clusters: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Assign topic labels to questions and return summaries.

        Args:
            df: Questions dataframe.
            embeddings: Embedding matrix.
            n_clusters: Optional fixed cluster count.

        Returns:
            Tuple of (annotated questions dataframe, topic summary dataframe).
        """
        labels = self.fit(embeddings, n_clusters=n_clusters)
        summary = self.build_topic_summary(df, labels)
        label_map = summary.set_index("topic_id")["topic_label"].to_dict()

        annotated = df.copy()
        annotated["topic_id"] = labels
        annotated["topic_label"] = annotated["topic_id"].map(label_map)
        return annotated, summary

    @staticmethod
    def identify_trends(summary_df: pd.DataFrame) -> pd.DataFrame:
        """Identify trending and declining topics across years.

        Args:
            summary_df: Topic summary with frequency_by_year dictionaries.

        Returns:
            Dataframe with trend labels.
        """
        records: list[dict[str, Any]] = []
        for _, row in summary_df.iterrows():
            freq: dict[Any, int] = row["frequency_by_year"]
            if not freq:
                records.append({**row.to_dict(), "trend": "stable"})
                continue

            years = sorted(freq.keys())
            if len(years) < 2:
                trend = "stable"
            else:
                first_half = sum(freq[y] for y in years[: len(years) // 2])
                second_half = sum(freq[y] for y in years[len(years) // 2 :])
                if second_half > first_half * 1.2:
                    trend = "trending"
                elif second_half < first_half * 0.8:
                    trend = "declining"
                else:
                    trend = "stable"

            record = row.to_dict()
            record["trend"] = trend
            records.append(record)

        return pd.DataFrame(records)
