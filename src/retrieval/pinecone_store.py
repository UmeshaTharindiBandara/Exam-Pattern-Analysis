"""Pinecone vector store helpers for exam content."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.embeddings.embedder import QuestionEmbedder
from src.utils import setup_logging

load_dotenv()
logger = setup_logging(__name__)


@dataclass
class PineconeMatch:
    """Normalized Pinecone match payload."""

    id: str
    score: float
    text: str
    namespace: str
    metadata: dict[str, Any]


class PineconeVectorStore:
    """Store and retrieve embeddings from Pinecone."""

    def __init__(
        self,
        index_name: str | None = None,
        api_key: str | None = None,
        embedder: QuestionEmbedder | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("PINECONE_API_KEY", "")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "quickstart")
        self._client: Any | None = None
        self._index: Any | None = None
        self._embedder: QuestionEmbedder | None = embedder

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ValueError(
                "PINECONE_API_KEY is required for retrieval. Set it in .env or the environment."
            )

        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise ValueError("pinecone is not installed. Run: pip install pinecone") from exc

        self._client = Pinecone(api_key=self.api_key)
        return self._client

    def _get_index(self) -> Any:
        if self._index is not None:
            return self._index

        client = self._get_client()
        self._index = client.Index(self.index_name)
        return self._index

    @staticmethod
    def _to_vector(values: np.ndarray | list[float]) -> list[float]:
        return np.asarray(values, dtype=np.float32).tolist()

    def _get_embedder(self) -> QuestionEmbedder:
        if self._embedder is None:
            self._embedder = QuestionEmbedder()
        return self._embedder

    def upsert_dataframe(
        self,
        df: pd.DataFrame,
        *,
        text_column: str,
        namespace: str,
        id_prefix: str,
        metadata_columns: list[str] | None = None,
        embeddings: np.ndarray | None = None,
        force: bool = True,
    ) -> int:
        """Embed and upsert a dataframe into Pinecone."""
        if df.empty:
            return 0

        if embeddings is None:
            embedder = self._get_embedder()
            embeddings = embedder.encode_dataframe(
                df,
                text_column=text_column,
                cache_key=f"{namespace}_{len(df)}",
                force_recompute=force,
            )

        index = self._get_index()
        vectors = []
        metadata_columns = metadata_columns or []
        for row_index, (_, row) in enumerate(df.iterrows()):
            text_value = str(row.get(text_column, "")).strip()
            if not text_value:
                continue

            metadata: dict[str, Any] = {
                "text": text_value,
                "content_type": namespace,
            }
            for column in metadata_columns:
                value = row.get(column)
                if pd.notna(value):
                    metadata[column] = value.item() if hasattr(value, "item") else value

            vector_id = row.get("vector_id") or f"{id_prefix}-{row_index}"
            vectors.append(
                {
                    "id": str(vector_id),
                    "values": self._to_vector(embeddings[row_index]),
                    "metadata": metadata,
                }
            )

        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
        return len(vectors)

    def query(
        self,
        query_text: str,
        *,
        namespace: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[PineconeMatch]:
        """Search Pinecone and return normalized matches."""
        if not query_text.strip():
            return []

        query_embedding = self._get_embedder().encode([query_text])[0]
        index = self._get_index()
        response = index.query(
            namespace=namespace,
            vector=self._to_vector(query_embedding),
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter,
        )

        matches: list[PineconeMatch] = []
        for match in getattr(response, "matches", []) or []:
            metadata = getattr(match, "metadata", None) or {}
            text = str(metadata.get("text", ""))
            matches.append(
                PineconeMatch(
                    id=str(getattr(match, "id", "")),
                    score=float(getattr(match, "score", 0.0)),
                    text=text,
                    namespace=namespace,
                    metadata=dict(metadata),
                )
            )
        return matches

    def query_multiple(
        self,
        query_text: str,
        *,
        namespaces: list[str],
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[PineconeMatch]:
        """Search across several namespaces and merge the results."""
        matches: list[PineconeMatch] = []
        for namespace in namespaces:
            matches.extend(
                self.query(
                    query_text,
                    namespace=namespace,
                    top_k=top_k,
                    metadata_filter=metadata_filter,
                )
            )
        return matches

    def delete_namespace(self, namespace: str) -> None:
        """Delete all vectors in one namespace."""
        try:
            index = self._get_index()
            index.delete(delete_all=True, namespace=namespace)
        except Exception as exc:
            logger.warning("Could not delete Pinecone namespace %s: %s", namespace, exc)
