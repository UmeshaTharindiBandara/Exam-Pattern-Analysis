"""LLM-powered exam question generation with Mistral chat completions."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

from dotenv import load_dotenv

from src.utils import setup_logging

load_dotenv()
logger = setup_logging(__name__)

DifficultyLevel = Literal["Easy", "Medium", "Hard"]
PromptStrategy = Literal["direct", "chain_of_thought", "context_aware"]


class QuestionGenerator:
    """Generate probable future exam questions using the Mistral API."""

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        """Initialize the question generator.

        Args:
            api_key: Optional API key override.
        """
        self.api_key = api_key

    def _resolve_api_key(self) -> str:
        """Resolve API key from override or environment."""
        if self.api_key:
            return self.api_key

        key = os.getenv("MISTRAL_API_KEY", "")
        if not key or key == "your_key_here":
            raise ValueError(
                "Mistral API key not configured. Set MISTRAL_API_KEY in .env or enter it in the sidebar."
            )
        return key

    def _build_prompt(
        self,
        topic: str,
        num_questions: int,
        difficulty: DifficultyLevel,
        strategy: PromptStrategy,
        subject: str | None = None,
        sample_questions: list[str] | None = None,
        subject_material: str | None = None,
    ) -> str:
        """Build prompt text for the selected strategy.

        Args:
            topic: Discovered topic label from uploaded exam analysis.
            num_questions: Number of questions to generate.
            difficulty: Difficulty level.
            strategy: Prompting strategy.
            subject: User-defined subject name.
            sample_questions: Past exam questions for context-aware mode.
            subject_material: Reference text from subject/syllabus PDFs.

        Returns:
            Prompt string.
        """
        subject_line = f"Subject: {subject}\n" if subject else ""
        material_block = ""
        if subject_material:
            material_block = (
                "Use this subject reference material (syllabus/notes) for scope and terminology:\n"
                f"{subject_material}\n\n"
            )

        base = (
            f"{subject_line}"
            f"Generate {num_questions} original university exam questions on the topic "
            f"'{topic}' at {difficulty} difficulty."
        )
        schema = (
            "Return ONLY valid JSON as a list of objects with keys: "
            "question, topic, difficulty, marks, type."
        )

        if strategy == "direct":
            return (
                f"{material_block}{base}\n"
                "Use varied question types (MCQ, short answer, essay, calculation).\n"
                f"{schema}"
            )

        if strategy == "chain_of_thought":
            return (
                f"{material_block}"
                f"Analyze historical exam patterns for subject '{subject or 'general'}' "
                f"and topic '{topic}'. Identify common themes, then generate "
                f"{num_questions} {difficulty} questions.\n"
                "Think step-by-step internally, but output only the final JSON.\n"
                f"{schema}"
            )

        context = "\n".join(f"- {q}" for q in (sample_questions or [])[:8])
        return (
            f"{material_block}{base}\n"
            "Use these real past exam questions as style and scope references:\n"
            f"{context if context else '- No prior exam samples available for this topic.'}\n"
            "Match academic tone and complexity while creating new original questions.\n"
            f"{schema}"
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the Gemini API.

        Args:
            prompt: User prompt.

        Returns:
            Raw model response text.
        """
        api_key = self._resolve_api_key()

        try:
            from mistralai.client import Mistral
        except ImportError as exc:
            raise ValueError(
                "mistralai is not installed. Run: pip install mistralai"
            ) from exc

        model_name = os.getenv("MISTRAL_CHAT_MODEL", "mistral-medium-latest")
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        return "".join(
            chunk.text if hasattr(chunk, "text") else str(chunk)
            for chunk in content
        )

    @staticmethod
    def _parse_json_response(raw_text: str) -> list[dict[str, Any]]:
        """Parse JSON list from LLM response.

        Args:
            raw_text: Raw LLM output.

        Returns:
            Parsed question dictionaries.
        """
        text = raw_text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            text = text[start : end + 1]

        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("questions", [])
        if not isinstance(data, list):
            raise ValueError("LLM response is not a JSON list.")

        normalized: list[dict[str, Any]] = []
        for item in data:
            normalized.append(
                {
                    "question": str(item.get("question", "")).strip(),
                    "topic": str(item.get("topic", "")).strip(),
                    "difficulty": str(item.get("difficulty", "Medium")).strip(),
                    "marks": int(item.get("marks", 5)),
                    "type": str(item.get("type", "short_answer")).strip(),
                }
            )
        return [q for q in normalized if q["question"]]

    def generate(
        self,
        topic: str,
        num_questions: int = 5,
        difficulty: DifficultyLevel = "Medium",
        strategy: PromptStrategy = "direct",
        subject: str | None = None,
        sample_questions: list[str] | None = None,
        subject_material: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate structured exam questions via the Gemini API.

        Args:
            topic: Topic to generate questions for.
            num_questions: Number of questions.
            difficulty: Difficulty level.
            strategy: Prompting strategy.
            subject: User-defined subject name.
            sample_questions: Optional past exam questions for context.
            subject_material: Optional syllabus/reference text.

        Returns:
            List of generated question dictionaries.

        Raises:
            ValueError: If API key is missing or LLM response is invalid.
        """
        prompt = self._build_prompt(
            topic=topic,
            num_questions=num_questions,
            difficulty=difficulty,
            strategy=strategy,
            subject=subject,
            sample_questions=sample_questions,
            subject_material=subject_material,
        )

        raw = self._call_llm(prompt)
        questions = self._parse_json_response(raw)
        if not questions:
            raise ValueError("Gemini returned no valid questions. Check your API key and try again.")
        return questions[:num_questions]