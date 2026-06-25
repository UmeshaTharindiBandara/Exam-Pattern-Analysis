"""Quick validation script for project imports (requires uploaded PDF data for full test)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Validate core module imports and syntax."""
    import py_compile

    from src.pipeline import NoQuestionDataError, load_questions
    from src.preprocessing.text_cleaner import TextCleaner

    py_compile.compile(str(PROJECT_ROOT / "app" / "streamlit_app.py"), doraise=True)
    print("Streamlit app syntax check passed.")

    try:
        load_questions()
        print("Found uploaded exam data — run full pipeline test manually after PDF upload.")
    except NoQuestionDataError:
        print("No uploaded exam data yet (expected before first PDF upload).")

    cleaner = TextCleaner()
    sample = cleaner.enrich_dataframe(
        __import__("pandas").DataFrame(
            [{"question_text": "Explain the TCP/IP protocol stack and its layers."}]
        )
    )
    assert "cleaned_text" in sample.columns
    assert "question_type" in sample.columns
    print("NLP pipeline check passed.")
    print("Validation successful.")


if __name__ == "__main__":
    main()
