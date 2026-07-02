"""Quick diagnostic for Mistral API key and OCR endpoint reachability."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        print("ERROR: MISTRAL_API_KEY missing")
        return 2

    try:
        from mistralai import Mistral
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: mistralai import failed: {exc}")
        return 3

    try:
        client = Mistral(api_key=api_key)
        # Small authenticated call to validate key and connectivity.
        models = client.models.list()
        count = len(getattr(models, "data", []) or [])
        print(f"OK: authenticated with Mistral. models={count}")
        return 0
    except Exception as exc:
        print(f"ERROR: Mistral auth/request failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
