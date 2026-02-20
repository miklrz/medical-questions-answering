"""
LLM generation with structured output: try native, fallback to JSON parsing.
"""

import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

from src.api.schemas import MedicalAnswer


def get_llm():
    """Get LLM instance."""
    return ChatOpenAI(
        model="mistral-7b-instruct-v0.1",
        temperature=0.0,
        base_url=f"http://{os.getenv('HOST')}:1234/v1",
        api_key="not-needed",
        timeout=30.0,
    )


def extract_json(text: str) -> dict | None:
    """Extract JSON object from raw LLM output."""
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Greedy: find first { and last }
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def parse_to_medical_answer(raw: dict, sources: list[str]) -> MedicalAnswer:
    """Parse dict to MedicalAnswer with validation."""
    answer = str(raw.get("answer", "")) or "Извините, не удалось сформировать ответ."
    confidence = float(raw.get("confidence", 0.8))
    confidence = max(0.0, min(1.0, confidence))
    requires_doctor = bool(raw.get("requires_doctor_visit", False))
    warnings = raw.get("warnings", [])
    if isinstance(warnings, str):
        warnings = [warnings] if warnings else []
    return MedicalAnswer(
        answer=answer,
        confidence=confidence,
        sources=sources,
        requires_doctor_visit=requires_doctor,
        warnings=warnings,
    )


def generate_structured(prompt: str, sources: list[str]) -> MedicalAnswer:
    """
    Generate MedicalAnswer from prompt.
    Uses JSON parsing fallback for models without function calling.
    """
    llm = get_llm()

    # Try native structured output first (OpenAI, some local models)
    try:
        structured_llm = llm.with_structured_output(MedicalAnswer)
        result = structured_llm.invoke([HumanMessage(content=prompt)])
        result.sources = sources
        return result
    except Exception:
        pass

    # Fallback: raw invoke + JSON parse
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        data = extract_json(response)
        if data:
            return parse_to_medical_answer(data, sources)
    except Exception:
        pass

    # Final fallback
    return MedicalAnswer(
        answer="Извините, не удалось сформировать ответ. Обратитесь к врачу.",
        confidence=0.0,
        sources=sources,
        requires_doctor_visit=True,
        warnings=["Ответ не является медицинской консультацией."],
    )
