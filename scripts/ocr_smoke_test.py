"""Smoke test for Mistral OCR with a generated one-page PDF."""

from __future__ import annotations

from pathlib import Path
import sys

from fpdf import FPDF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.mistral_ocr import MistralOCRClient


def main() -> None:
    pdf_path = Path("data/raw/_ocr_smoke.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Q1. What is AI?", ln=True)
    pdf.output(str(pdf_path))

    client = MistralOCRClient()
    pages = client.document_to_pages(pdf_path)
    print(f"pages={len(pages)}")
    if pages:
        print(pages[0]["content_text"][:200])


if __name__ == "__main__":
    main()
