"""Extract and segment exam questions from PDF documents using Mistral OCR."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.ocr.mistral_ocr import MistralOCRClient
from src.utils import PROCESSED_DIR, setup_logging

logger = setup_logging(__name__)

QUESTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:Q(?:uestion)?\.?\s*)?(?P<num>\d{1,3})[\.\):]\s*)"
    r"(?P<text>.+?)"
    r"(?=(?:\n\s*(?:Q(?:uestion)?\.?\s*)?\d{1,3}[\.\):])|\Z)",
    re.DOTALL | re.IGNORECASE,
)
MARKS_PATTERN = re.compile(
    r"\((\d{1,3})\s*(?:marks?|pts?|points?)\)",
    re.IGNORECASE,
)
HEADER_FOOTER_PATTERN = re.compile(
    r"(page\s+\d+|exam\s+paper|university|department|faculty|"
    r"confidential|instructions|time allowed|total marks)",
    re.IGNORECASE,
)
PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)


class PDFExtractor:
    """Extract exam questions and lecture notes from PDFs via Mistral OCR."""

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize the PDF extractor.

        Args:
            output_dir: Directory for processed CSV output.
        """
        self.output_dir = output_dir or PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_client = MistralOCRClient()

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract OCR text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Combined OCR text content.

        Raises:
            ValueError: If the OCR request fails or the PDF yields no readable text.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            page_chunks = self.ocr_client.document_to_pages(pdf_path)
        except Exception as exc:
            logger.exception("Failed to OCR PDF: %s", pdf_path)
            raise ValueError(
                f"Unable to OCR PDF '{pdf_path.name}'. Check your Mistral API key or file contents."
            ) from exc

        text_parts = [chunk["content_text"] for chunk in page_chunks if chunk.get("content_text")]
        combined = "\n\n".join(text_parts).strip()
        if len(combined) < 20:
            raise ValueError(
                f"PDF '{pdf_path.name}' appears to contain insufficient readable OCR text."
            )
        return combined

    def clean_text(self, text: str) -> str:
        """Remove headers, footers, and page numbers from extracted text.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text.
        """
        cleaned = PAGE_NUMBER_PATTERN.sub("", text)
        lines = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if HEADER_FOOTER_PATTERN.search(stripped) and len(stripped.split()) <= 8:
                continue
            lines.append(stripped)
        return "\n".join(lines)

    def segment_questions(self, text: str) -> list[dict[str, Any]]:
        """Segment cleaned text into individual questions.

        Args:
            text: Cleaned exam paper text.

        Returns:
            List of question dictionaries with text and optional marks.
        """
        questions: list[dict[str, Any]] = []
        matches = list(QUESTION_PATTERN.finditer(text))

        if not matches:
            fallback_blocks = [
                block.strip()
                for block in re.split(r"\?\s*(?=\n|$)", text)
                if block.strip()
            ]
            for idx, block in enumerate(fallback_blocks, start=1):
                question_text = block if block.endswith("?") else f"{block}?"
                questions.append(
                    {
                        "question_id": f"Q{idx}",
                        "question_text": question_text.strip(),
                        "marks": self._extract_marks(question_text),
                    }
                )
            return questions

        for match in matches:
            question_num = match.group("num")
            question_text = match.group("text").strip()
            question_text = re.sub(r"\s+", " ", question_text)
            if len(question_text) < 10:
                continue
            questions.append(
                {
                    "question_id": f"Q{question_num}",
                    "question_text": question_text,
                    "marks": self._extract_marks(question_text),
                }
            )
        return questions

    @staticmethod
    def _build_chunk_id(pdf_path: Path, page_index: int) -> str:
        return f"{pdf_path.stem}_page_{page_index + 1:03d}"

    def _extract_marks(self, text: str) -> int | None:
        """Extract marks allocation from question text.

        Args:
            text: Question text.

        Returns:
            Marks value if found, otherwise None.
        """
        match = MARKS_PATTERN.search(text)
        if match:
            return int(match.group(1))
        return None

    def process_pdf(
        self,
        pdf_path: Path,
        subject: str,
        year: int,
    ) -> pd.DataFrame:
        """Extract, clean, segment, and structure questions from a PDF.

        Args:
            pdf_path: Path to uploaded PDF.
            subject: Subject name for the exam paper.
            year: Exam year.

        Returns:
            DataFrame with question records.
        """
        raw_text = self.extract_text_from_pdf(pdf_path)
        cleaned_text = self.clean_text(raw_text)
        segmented = self.segment_questions(cleaned_text)

        if not segmented:
            raise ValueError(
                f"No questions detected in '{pdf_path.name}'. "
                "Try a text-based PDF with numbered questions."
            )

        records = []
        for idx, item in enumerate(segmented, start=1):
            records.append(
                {
                    "question_id": item.get("question_id") or f"Q{idx}",
                    "question_text": item["question_text"],
                    "year": year,
                    "subject": subject,
                    "marks": item.get("marks"),
                    "source_file": pdf_path.name,
                    "content_type": "past-paper",
                }
            )
        return pd.DataFrame(records)

    def save_questions(
        self,
        df: pd.DataFrame,
        filename: str = "extracted_questions.csv",
    ) -> Path:
        """Save extracted questions to CSV.

        Args:
            df: Question dataframe.
            filename: Output filename.

        Returns:
            Path to saved CSV file.
        """
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info("Saved %s questions to %s", len(df), output_path)
        return output_path

    def process_and_save(
        self,
        pdf_path: Path,
        subject: str,
        year: int,
        filename: str = "extracted_questions.csv",
        append: bool = True,
    ) -> pd.DataFrame:
        """Process a PDF and optionally append results to existing CSV.

        Args:
            pdf_path: PDF file path.
            subject: Subject label.
            year: Exam year.
            filename: CSV filename.
            append: Whether to append to existing processed data.

        Returns:
            Combined dataframe after save.
        """
        new_df = self.process_pdf(pdf_path, subject=subject, year=year)
        output_path = self.output_dir / filename

        if append and output_path.exists():
            existing = pd.read_csv(output_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.drop_duplicates(
            subset=["question_text", "year", "subject"],
            keep="last",
            inplace=True,
        )
        combined.to_csv(output_path, index=False)
        return combined

    def process_subject_pdf(self, pdf_path: Path, subject: str) -> pd.DataFrame:
        """Extract reference text from a subject/syllabus PDF.

        Args:
            pdf_path: Path to subject material PDF.
            subject: User-defined subject name (any discipline).

        Returns:
            DataFrame with subject reference content.
        """
        try:
            page_chunks = self.ocr_client.document_to_pages(pdf_path)
        except Exception as exc:
            logger.exception("Failed to OCR subject PDF: %s", pdf_path)
            raise ValueError(
                f"Unable to OCR subject PDF '{pdf_path.name}'. Check your Mistral API key or file contents."
            ) from exc

        records = []
        for chunk in page_chunks:
            cleaned_text = self.clean_text(chunk["content_text"])
            if not cleaned_text:
                continue
            page_index = int(chunk.get("page_index", 0))
            records.append(
                {
                    "subject": subject.strip(),
                    "source_file": pdf_path.name,
                    "page_index": page_index,
                    "chunk_id": self._build_chunk_id(pdf_path, page_index),
                    "content_type": "lecture-pdf",
                    "content_text": cleaned_text,
                }
            )

        if not records:
            raise ValueError(
                f"No readable OCR chunks were produced from '{pdf_path.name}'."
            )

        return pd.DataFrame(records)
