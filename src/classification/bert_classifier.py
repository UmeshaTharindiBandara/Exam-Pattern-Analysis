"""BERT-based question type classification using HuggingFace Transformers."""

from __future__ import annotations

from typing import Literal

from src.utils import setup_logging

logger = setup_logging(__name__)

QuestionType = Literal["MCQ", "short_answer", "essay", "calculation", "unknown"]


class BERTQuestionClassifier:
    """Classify exam questions using a DistilBERT zero-shot model."""

    def __init__(self, model_name: str = "typeform/distilbert-base-uncased-mnli") -> None:
        """Initialize the classifier.

        Args:
            model_name: HuggingFace model name for zero-shot classification.
        """
        self.model_name = model_name
        self._pipeline = None
        self._labels = ["MCQ", "short answer", "essay", "calculation"]

    @property
    def pipeline(self):
        """Lazy-load the transformers zero-shot pipeline."""
        if self._pipeline is None:
            from transformers import pipeline

            logger.info("Loading BERT classifier model: %s", self.model_name)
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
            )
        return self._pipeline

    def classify(self, text: str) -> QuestionType:
        """Classify a question using BERT zero-shot labels.

        Args:
            text: Question text.

        Returns:
            Predicted question type.
        """
        try:
            result = self.pipeline(text, candidate_labels=self._labels)
            label = result["labels"][0].lower().replace(" ", "_")
            if label == "short_answer":
                return "short_answer"
            if label in {"mcq", "essay", "calculation"}:
                return label  # type: ignore[return-value]
            return "unknown"
        except Exception as exc:
            logger.warning("BERT classification failed, returning unknown: %s", exc)
            return "unknown"
