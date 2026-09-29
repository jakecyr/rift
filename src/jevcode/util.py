"""Small helpers shared by the agent."""

from __future__ import annotations

import json
import re


def clip(text: str, limit: int) -> str:
    if limit < 1 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[truncated {omitted} chars]"


def extract_json(text: str) -> dict:
    """Pull one JSON object out of a model reply."""
    cleaned: str = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("model reply did not contain a JSON object") from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError(f"model reply was not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("model reply JSON was not an object")
    return value
