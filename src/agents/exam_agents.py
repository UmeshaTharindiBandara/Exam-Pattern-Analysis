"""Multi-agent orchestration for exam paper analysis and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.evaluation.retrieval_metrics import RetrievalMetricSummary, compute_retrieval_metrics
from src.generation.question_generator import QuestionGenerator
from src.pipeline import get_subject_context
from src.preprocessing.pdf_extractor import PDFExtractor
from src.retrieval.pinecone_store import PineconeVectorStore
from src.retrieval.reranker import QuestionReranker
from src.utils import setup_logging

load_dotenv()
logger = setup_logging(__name__)


@dataclass
class RetrievalBundle:
    """Context returned to the prediction agent."""

    matches: pd.DataFrame
    sample_questions: list[str]
    subject_context: str


class PastPaperAgent:
    """Analyze uploaded exam papers and prepare them for retrieval."""

    def __init__(self, extractor: PDFExtractor | None = None) -> None:
        self.extractor = extractor or PDFExtractor()

    def ingest(self, pdf_path, subject: str, year: int) -> pd.DataFrame:
        return self.extractor.process_pdf(pdf_path, subject=subject, year=year)


class LecturePdfAgent:
    """Analyze lecture PDFs and prepare their chunks for retrieval."""

    def __init__(self, extractor: PDFExtractor | None = None) -> None:
        self.extractor = extractor or PDFExtractor()

    def ingest(self, pdf_path, subject: str) -> pd.DataFrame:
        return self.extractor.process_subject_pdf(pdf_path, subject=subject)


class PredictionAgent:
    """Retrieve evidence and generate probable future questions."""

    def __init__(
        self,
        vector_store: PineconeVectorStore | None = None,
        reranker: QuestionReranker | None = None,
        generator: QuestionGenerator | None = None,
    ) -> None:
        self.vector_store = vector_store or PineconeVectorStore()
        self.reranker = reranker or QuestionReranker()
        self.generator = generator or QuestionGenerator()

    def retrieve_context(
        self,
        *,
        query: str,
        subject: str,
        questions_df: pd.DataFrame,
        materials_df: pd.DataFrame,
        top_k: int = 8,
    ) -> RetrievalBundle:
        metadata_filter = {"subject": {"$eq": subject}} if subject else None
        matches = self.vector_store.query_multiple(
            query,
            namespaces=["past-papers", "lecture-pdfs"],
            top_k=max(20, top_k * 2),
            metadata_filter=metadata_filter,
        )

        if not matches:
            fallback = pd.DataFrame(columns=["text", "score", "namespace", "metadata"])
            subject_context = get_subject_context(subject, materials_df)
            if subject and "subject" in questions_df.columns:
                subject_rows = questions_df[questions_df["subject"].astype(str).str.lower() == subject.lower()]
            else:
                subject_rows = questions_df
            sample_questions = subject_rows["question_text"].head(top_k).tolist() if "question_text" in subject_rows.columns else []
            return RetrievalBundle(fallback, sample_questions, subject_context)

        candidate_rows = []
        for match in matches:
            candidate_rows.append(
                {
                    "id": match.id,
                    "text": match.text,
                    "score": match.score,
                    "namespace": match.namespace,
                    "metadata": match.metadata,
                    "subject": match.metadata.get("subject", subject),
                    "source_file": match.metadata.get("source_file", ""),
                    "topic_label": match.metadata.get("topic_label", ""),
                    "question_type": match.metadata.get("question_type", ""),
                    "content_type": match.metadata.get("content_type", match.namespace),
                }
            )

        candidate_df = pd.DataFrame(candidate_rows)
        reranked = self.reranker.rerank(query, candidate_df, top_k=top_k)

        subject_context = get_subject_context(subject, materials_df)
        sample_questions = reranked[reranked["content_type"] == "past-papers"]["text"].head(top_k).tolist()
        if not sample_questions:
            sample_questions = reranked["text"].head(top_k).tolist()

        return RetrievalBundle(reranked, sample_questions, subject_context)

    def generate(
        self,
        *,
        topic: str,
        subject: str,
        num_questions: int,
        difficulty: str,
        strategy: str,
        questions_df: pd.DataFrame,
        topics_df: pd.DataFrame,
        materials_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        retrieval = self.retrieve_context(
            query=topic,
            subject=subject,
            questions_df=questions_df,
            materials_df=materials_df,
        )

        generated = self.generator.generate(
            topic=topic,
            num_questions=num_questions,
            difficulty=difficulty,  # type: ignore[arg-type]
            strategy=strategy,  # type: ignore[arg-type]
            subject=subject,
            sample_questions=retrieval.sample_questions,
            subject_material=retrieval.subject_context or None,
        )
        return generated


class EvaluationAgent:
    """Run retrieval evaluation over the indexed corpus."""

    def __init__(self, prediction_agent: PredictionAgent | None = None) -> None:
        self.prediction_agent = prediction_agent or PredictionAgent()

    def evaluate_retrieval(
        self,
        questions_df: pd.DataFrame,
        materials_df: pd.DataFrame,
        *,
        subject: str | None = None,
        sample_size: int = 25,
        top_k: int = 5,
    ) -> tuple[RetrievalMetricSummary, pd.DataFrame]:
        if questions_df.empty:
            return RetrievalMetricSummary(0.0, 0.0, 0.0, 0.0, 0), pd.DataFrame()

        if subject and "subject" in questions_df.columns:
            working = questions_df[questions_df["subject"].astype(str).str.lower() == subject.lower()].copy()
        else:
            working = questions_df.copy()

        if working.empty:
            return RetrievalMetricSummary(0.0, 0.0, 0.0, 0.0, 0), pd.DataFrame()

        working = working.sample(min(sample_size, len(working)), random_state=42).reset_index(drop=True)

        rows = []
        ranked_labels: list[list[int]] = []
        for _, row in working.iterrows():
            row_subject = subject or str(row.get("subject", ""))
            retrieval = self.prediction_agent.retrieve_context(
                query=str(row.get("question_text", "")),
                subject=row_subject,
                questions_df=questions_df,
                materials_df=materials_df,
                top_k=top_k,
            )

            if retrieval.matches.empty:
                ranked_labels.append([])
                continue

            relevance = []
            for _, candidate in retrieval.matches.iterrows():
                same_topic = str(candidate.get("topic_label", "")) == str(row.get("topic_label", ""))
                same_text = str(candidate.get("text", "")).strip() == str(row.get("question_text", "")).strip()
                relevance.append(int(same_topic and not same_text))
                rows.append(
                    {
                        "query_question": row.get("question_text", ""),
                        "retrieved_text": candidate.get("text", ""),
                        "rank_score": candidate.get("rerank_score", candidate.get("score", 0.0)),
                        "content_type": candidate.get("content_type", ""),
                        "topic_label": candidate.get("topic_label", ""),
                        "query_topic": row.get("topic_label", ""),
                        "is_relevant": relevance[-1],
                    }
                )
            ranked_labels.append(relevance)

        summary = compute_retrieval_metrics(ranked_labels, k=top_k)
        details = pd.DataFrame(rows)
        return summary, details


class ExamAgentSystem:
    """Coordinates the specialist agents used by the app."""

    def __init__(self) -> None:
        self.vector_store = PineconeVectorStore()
        self.reranker = QuestionReranker()
        self.generator = QuestionGenerator()
        self.paper_agent = PastPaperAgent()
        self.lecture_agent = LecturePdfAgent()
        self.prediction_agent = PredictionAgent(
            vector_store=self.vector_store,
            reranker=self.reranker,
            generator=self.generator,
        )
        self.evaluation_agent = EvaluationAgent(self.prediction_agent)

    def index_corpus(
        self,
        questions_df: pd.DataFrame,
        materials_df: pd.DataFrame,
        *,
        question_embeddings=None,
    ) -> dict[str, int]:
        """Upsert the local corpus into Pinecone."""
        indexed = {"questions": 0, "materials": 0}

        try:
            indexed["questions"] = self.vector_store.upsert_dataframe(
                questions_df,
                text_column="question_text",
                namespace="past-papers",
                id_prefix="question",
                metadata_columns=["subject", "year", "marks", "source_file", "question_id", "topic_label", "question_type"],
                embeddings=question_embeddings,
            )
        except Exception as exc:
            logger.warning("Could not sync questions to Pinecone: %s", exc)

        try:
            if not materials_df.empty:
                indexed["materials"] = self.vector_store.upsert_dataframe(
                    materials_df,
                    text_column="content_text",
                    namespace="lecture-pdfs",
                    id_prefix="material",
                    metadata_columns=["subject", "source_file", "page_index", "chunk_id"],
                )
        except Exception as exc:
            logger.warning("Could not sync lecture PDFs to Pinecone: %s", exc)

        return indexed
