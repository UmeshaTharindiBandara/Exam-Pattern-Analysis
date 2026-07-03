"""Pipeline helpers for loading and analyzing exam question data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.classification.topic_classifier import TopicClassifier
from src.embeddings.embedder import QuestionEmbedder
from src.preprocessing.text_cleaner import TextCleaner
from src.utils import PROCESSED_DIR

QUESTIONS_CSV = PROCESSED_DIR / "questions.csv"
SUBJECT_MATERIALS_CSV = PROCESSED_DIR / "subject_materials.csv"


class NoQuestionDataError(FileNotFoundError):
    """Raised when no exam paper data has been uploaded yet."""


def load_questions(path: Path | None = None) -> pd.DataFrame:
    """Load questions extracted from uploaded exam paper PDFs.

    Args:
        path: Optional explicit CSV path.

    Returns:
        Questions dataframe.

    Raises:
        NoQuestionDataError: If no uploaded exam data exists yet.
    """
    csv_path = path or QUESTIONS_CSV
    if not csv_path.exists():
        raise NoQuestionDataError(
            "No exam papers found. Upload past exam PDFs on the Upload & Process page "
            "to begin analysis."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise NoQuestionDataError(
            "The questions file is empty. Upload past exam PDFs to extract questions."
        )
    return df


def load_subject_materials(path: Path | None = None) -> pd.DataFrame:
    """Load reference text extracted from subject/syllabus PDFs.

    Args:
        path: Optional explicit CSV path.

    Returns:
        Subject materials dataframe, or empty dataframe if none uploaded.
    """
    csv_path = path or SUBJECT_MATERIALS_CSV
    if not csv_path.exists():
        return pd.DataFrame(
            columns=[
                "subject",
                "source_file",
                "page_index",
                "chunk_id",
                "content_type",
                "content_text",
            ]
        )

    df = pd.read_csv(csv_path)
    for column in ["page_index", "chunk_id", "content_type"]:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def get_subjects(df: pd.DataFrame) -> list[str]:
    """Return sorted unique subject names from a questions dataframe.

    Args:
        df: Questions dataframe.

    Returns:
        Sorted list of subject names.
    """
    if "subject" not in df.columns or df.empty:
        return []
    return sorted(df["subject"].dropna().astype(str).unique().tolist())


def get_subject_context(subject: str, materials_df: pd.DataFrame, max_chars: int = 4000) -> str:
    """Build subject reference context from uploaded syllabus/material PDFs.

    Args:
        subject: Subject name to look up.
        materials_df: Subject materials dataframe.
        max_chars: Maximum characters to include in context.

    Returns:
        Combined reference text for the subject.
    """
    if materials_df.empty or "subject" not in materials_df.columns:
        return ""

    subject_rows = materials_df[
        materials_df["subject"].astype(str).str.lower() == subject.lower()
    ]
    if subject_rows.empty:
        return ""

    chunks = subject_rows["content_text"].fillna("").astype(str).tolist()
    combined = "\n\n".join(chunks).strip()
    if len(combined) > max_chars:
        return combined[:max_chars] + "..."
    return combined


def filter_by_subject(df: pd.DataFrame, subject: str | None) -> pd.DataFrame:
    """Filter a dataframe to one subject, or return all rows if subject is None.

    Args:
        df: Input dataframe with a subject column.
        subject: Subject name, or None for all subjects.

    Returns:
        Filtered dataframe.
    """
    if not subject or "subject" not in df.columns:
        return df
    return df[df["subject"].astype(str).str.lower() == subject.lower()].copy()


def run_analysis_pipeline(
    df: pd.DataFrame,
    force_recompute: bool = False,
    cache_key: str = "exam_questions",
    embedder: QuestionEmbedder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Run NLP enrichment, embedding, and topic clustering pipeline.

    Topics are discovered dynamically from uploaded exam content via clustering,
    not from any predefined subject list.

    Args:
        df: Questions dataframe from uploaded PDFs.
        force_recompute: Force embedding recomputation.
        cache_key: Embedding cache identifier.
        embedder: Optional shared QuestionEmbedder instance. Callers that run this
            repeatedly (e.g. a long-lived app process) should pass in one cached
            instance rather than letting this function load its own copy of the
            embedding model every call.

    Returns:
        Tuple of (annotated questions, topic summary, embeddings).
    """
    if df.empty:
        raise ValueError("Cannot analyze an empty question set.")

    cleaner = TextCleaner()
    enriched = cleaner.enrich_dataframe(df)

    embedder = embedder or QuestionEmbedder()
    embeddings = embedder.encode_dataframe(
        enriched,
        text_column="question_text",
        cache_key=cache_key,
        force_recompute=force_recompute,
    )

    n = len(enriched)
    # Require at least ~4 questions per cluster so silhouette is meaningful.
    # n//4 ensures each cluster has enough members; floor at 2, cap at 10.
    max_k = min(10, max(2, n // 4))
    min_k = min(3, max(2, n // 6))
    classifier = TopicClassifier(min_clusters=min_k, max_clusters=max_k)
    annotated, summary = classifier.annotate_questions(enriched, embeddings)
    summary = classifier.identify_trends(summary)
    return annotated, summary, embeddings


def save_questions(df: pd.DataFrame) -> Path:
    """Persist combined exam questions to CSV.

    Args:
        df: Questions dataframe.

    Returns:
        Path to saved CSV.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(QUESTIONS_CSV, index=False)
    return QUESTIONS_CSV


def save_subject_materials(df: pd.DataFrame) -> Path:
    """Persist subject reference materials to CSV.

    Args:
        df: Subject materials dataframe.

    Returns:
        Path to saved CSV.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUBJECT_MATERIALS_CSV, index=False)
    return SUBJECT_MATERIALS_CSV


def append_questions(new_df: pd.DataFrame) -> pd.DataFrame:
    """Append newly extracted questions and deduplicate.

    Args:
        new_df: Newly extracted question records.

    Returns:
        Combined questions dataframe.
    """
    if QUESTIONS_CSV.exists():
        existing = pd.read_csv(QUESTIONS_CSV)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.drop_duplicates(
        subset=["question_text", "year", "subject"],
        keep="last",
        inplace=True,
    )
    save_questions(combined)
    return combined


def append_subject_materials(new_df: pd.DataFrame) -> pd.DataFrame:
    """Append subject reference material and deduplicate.

    Args:
        new_df: New subject material records.

    Returns:
        Combined subject materials dataframe.
    """
    if SUBJECT_MATERIALS_CSV.exists():
        existing = pd.read_csv(SUBJECT_MATERIALS_CSV)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    if "chunk_id" not in combined.columns:
        combined["chunk_id"] = combined["source_file"].astype(str)

    combined.drop_duplicates(
        subset=["subject", "source_file", "chunk_id"],
        keep="last",
        inplace=True,
    )
    save_subject_materials(combined)
    return combined
