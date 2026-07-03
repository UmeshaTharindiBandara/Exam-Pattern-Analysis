"""OCR-based text extraction for scanned / image-based PDF documents."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils import setup_logging

logger = setup_logging(__name__)

OCR_DPI = 200

_reader = None  # module-level EasyOCR Reader singleton (lazy-loaded)


def _get_reader():
    """Return the shared EasyOCR Reader, downloading models on first call (~200 MB)."""
    global _reader
    if _reader is None:
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError(
                "easyocr is not installed. Run: pip install easyocr"
            ) from exc
        logger.info("Loading EasyOCR model — first run downloads ~200 MB, please wait…")
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR model loaded successfully.")
    return _reader


def _render_pdf_to_images(pdf_path: Path, dpi: int = OCR_DPI) -> list:
    """Render each PDF page to a PIL Image using pypdfium2.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Rendering resolution — 200 DPI is a good accuracy/speed balance.

    Returns:
        List of PIL Image objects, one per page.
    """
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    images = []
    try:
        scale = dpi / 72.0  # PDF native DPI is 72
        for i in range(len(doc)):
            page = doc[i]
            bitmap = page.render(scale=scale, rotation=0)
            images.append(bitmap.to_pil())
    finally:
        doc.close()
    return images


def extract_text_with_ocr(pdf_path: Path, dpi: int = OCR_DPI) -> str:
    """Extract text from a scanned PDF using EasyOCR.

    Renders every page to an image via pypdfium2, runs EasyOCR (English) on
    each one, and joins the results.  The EasyOCR model is downloaded once and
    cached for subsequent calls.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Rendering DPI. Increase to 300 for very small print; decrease to
            150 to speed up processing on large documents.

    Returns:
        Full extracted text with pages separated by blank lines.

    Raises:
        ImportError: If easyocr is not installed.
        ValueError: If no text could be extracted from any page.
    """
    reader = _get_reader()

    logger.info("OCR: rendering '%s' at %d DPI", pdf_path.name, dpi)
    pages = _render_pdf_to_images(pdf_path, dpi=dpi)
    if not pages:
        raise ValueError(f"No pages could be rendered from '{pdf_path.name}'.")

    page_texts: list[str] = []
    for idx, img in enumerate(pages, start=1):
        gray = img.convert("L")  # grayscale improves OCR accuracy
        img_array = np.array(gray)
        try:
            results = reader.readtext(img_array, detail=0, paragraph=True)
            page_text = "\n".join(str(r) for r in results)
            if page_text.strip():
                page_texts.append(page_text)
            logger.debug(
                "OCR page %d/%d: %d chars extracted", idx, len(pages), len(page_text)
            )
        except Exception as exc:
            logger.warning("OCR failed on page %d of '%s': %s", idx, pdf_path.name, exc)

    if not page_texts:
        raise ValueError(
            f"OCR produced no text from '{pdf_path.name}'. "
            "The document may be encrypted, blank, or contain only graphics."
        )

    combined = "\n\n".join(page_texts)
    logger.info(
        "OCR complete for '%s': %d/%d pages yielded text, %d chars total",
        pdf_path.name,
        len(page_texts),
        len(pages),
        len(combined),
    )
    return combined
