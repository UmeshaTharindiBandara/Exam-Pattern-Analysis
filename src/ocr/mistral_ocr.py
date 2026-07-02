"""Mistral OCR helpers for PDF ingestion."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.utils import setup_logging

load_dotenv()
logger = setup_logging(__name__)


class MistralOCRClient:
    """Thin wrapper around the Mistral OCR endpoint."""

    def __init__(self, api_key: str | None = None, model: str = "mistral-ocr-latest") -> None:
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        self.model = model
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY is required for OCR. Set it in .env or the environment."
            )

        try:
            from mistralai import Mistral
        except ImportError as exc:
            raise ValueError(
                "mistralai is not installed. Run: pip install mistralai"
            ) from exc

        self._client = Mistral(api_key=self.api_key)
        return self._client

    @staticmethod
    def _encode_pdf(pdf_path: Path) -> str:
        with pdf_path.open("rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode("utf-8")

    def process_pdf(self, pdf_path: Path, include_blocks: bool = True) -> Any:
        """Run OCR on a local PDF using a base64 data URL."""
        encoded_pdf = self._encode_pdf(pdf_path)
        client = self._get_client()

        logger.info("Running Mistral OCR on %s", pdf_path.name)
        payload = {
            "model": self.model,
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{encoded_pdf}",
            },
            "include_image_base64": False,
        }

        if include_blocks:
            try:
                return client.ocr.process(
                    **payload,
                    include_blocks=True,
                )
            except TypeError:
                logger.warning(
                    "Installed mistralai SDK does not support include_blocks; retrying without it."
                )

        return client.ocr.process(**payload)

    @staticmethod
    def page_to_text(page: Any) -> str:
        """Convert one OCR page object to plain text."""
        if page is None:
            return ""

        markdown = getattr(page, "markdown", None)
        if markdown is None and isinstance(page, dict):
            markdown = page.get("markdown")
        if isinstance(markdown, str) and markdown.strip():
            return markdown.strip()

        blocks = getattr(page, "blocks", None)
        if blocks is None and isinstance(page, dict):
            blocks = page.get("blocks", [])

        text_parts: list[str] = []
        for block in blocks or []:
            content = getattr(block, "content", None)
            if content is None and isinstance(block, dict):
                content = block.get("content")
            if content:
                text_parts.append(str(content).strip())

        return "\n".join(part for part in text_parts if part).strip()

    def document_to_pages(self, pdf_path: Path) -> list[dict[str, Any]]:
        """Return normalized OCR page payloads for a PDF."""
        response = self.process_pdf(pdf_path)
        pages = getattr(response, "pages", None)
        if pages is None and isinstance(response, dict):
            pages = response.get("pages", [])

        normalized: list[dict[str, Any]] = []
        for index, page in enumerate(pages or []):
            text = self.page_to_text(page)
            if not text:
                continue
            normalized.append(
                {
                    "page_index": index,
                    "content_text": text,
                    "raw_page": page,
                }
            )
        return normalized
