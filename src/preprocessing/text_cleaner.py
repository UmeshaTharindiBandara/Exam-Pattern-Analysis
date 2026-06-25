"""NLP preprocessing and question type detection."""

from __future__ import annotations

import re
from typing import Literal

import nltk
import pandas as pd

from src.utils import setup_logging

logger = setup_logging(__name__)

QuestionType = Literal["MCQ", "short_answer", "essay", "calculation", "unknown"]

MCQ_PATTERN = re.compile(
    r"\b(?:which of the following|select the correct|choose the correct|"
    r"multiple choice|option [a-d]|options?:)\b",
    re.IGNORECASE,
)
ESSAY_PATTERN = re.compile(
    r"\b(?:discuss|explain in detail|critically evaluate|describe|"
    r"compare and contrast|write an essay|elaborate)\b",
    re.IGNORECASE,
)
CALCULATION_PATTERN = re.compile(
    r"\b(?:calculate|compute|derive|solve|find the value|"
    r"numerical|formula|equation|proof)\b",
    re.IGNORECASE,
)
SHORT_ANSWER_PATTERN = re.compile(
    r"\b(?:define|list|state|briefly|what is|name|identify)\b",
    re.IGNORECASE,
)


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources if missing."""
    resource_paths = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for resource, path in resource_paths.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:
                logger.warning("Could not download NLTK resource %s: %s", resource, exc)


class TextCleaner:
    """Clean and enrich exam question text using NLP techniques."""

    def __init__(self) -> None:
        """Initialize tokenizer, stopwords, and lemmatizer."""
        _ensure_nltk_resources()
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        self._word_tokenize = word_tokenize
        self._stopwords = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words.

        Args:
            text: Input text.

        Returns:
            List of tokens.
        """
        return self._word_tokenize(text.lower())

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove English stopwords from tokens.

        Args:
            tokens: Token list.

        Returns:
            Filtered tokens.
        """
        return [token for token in tokens if token.isalpha() and token not in self._stopwords]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Lemmatize tokens to base forms.

        Args:
            tokens: Token list.

        Returns:
            Lemmatized tokens.
        """
        return [self._lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text: str) -> str:
        """Apply full NLP preprocessing pipeline to text.

        Args:
            text: Raw question text.

        Returns:
            Cleaned and lemmatized text string.
        """
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        lemmas = self.lemmatize(tokens)
        return " ".join(lemmas)

    def detect_question_type(self, text: str) -> QuestionType:
        """Detect exam question type from textual cues.

        Args:
            text: Question text.

        Returns:
            Detected question type label.
        """
        if MCQ_PATTERN.search(text):
            return "MCQ"
        if CALCULATION_PATTERN.search(text):
            return "calculation"
        if ESSAY_PATTERN.search(text):
            return "essay"
        if SHORT_ANSWER_PATTERN.search(text) or len(text.split()) <= 25:
            return "short_answer"
        if len(text.split()) > 40:
            return "essay"
        return "unknown"

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cleaned text and question type columns to a dataframe.

        Args:
            df: Input questions dataframe.

        Returns:
            Enriched dataframe.
        """
        enriched = df.copy()
        enriched["cleaned_text"] = enriched["question_text"].apply(self.preprocess)
        enriched["question_type"] = enriched["question_text"].apply(self.detect_question_type)
        return enriched
