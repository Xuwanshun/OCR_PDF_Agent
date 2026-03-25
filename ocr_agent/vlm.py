from __future__ import annotations

import base64
import json
import os
from typing import Any, Literal


class VLMError(RuntimeError):
    pass


def image_file_to_base64_png(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise VLMError("openai package not available") from e
    return OpenAI()


def _extract_json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        # Strip fenced blocks if present: ```json\n{...}\n```
        first_nl = text.find("\n")
        last_fence = text.rfind("```")
        if first_nl != -1 and last_fence != -1 and last_fence > first_nl:
            text = text[first_nl + 1 : last_fence].strip()
    try:
        return json.loads(text)
    except Exception:
        # Best-effort: find first {...} span
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def extract_region(
    *,
    image_base64: str,
    region_type: Literal["table", "chart"],
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise VLMError("OPENAI_API_KEY is required for VLM extraction.")

    model = os.getenv("OCR_VISION_MODEL", "gpt-4o-mini")
    client = _get_openai_client()

    if region_type == "table":
        json_schema = {
            "headers": ["string"],
            "rows": [["any"]],
            "notes": "string|null",
            "confidence": "number|null",
            "units": "string|null",
        }
        instruction = (
            "You extract table data from an image.\n"
            "Return strict JSON only.\n"
            "Required fields: headers, rows, notes, confidence, units.\n"
            f"Schema example: {json.dumps(json_schema, ensure_ascii=False)}\n"
            "If uncertain, use null or empty arrays. Do not include extra text."
        )
    else:
        json_schema = {
            "chart_type": "string|null",
            "axes": {"x": "any", "y": "any"},
            "series": [{"name": "string", "points": [{"x": "any", "y": "any"}]}],
            "data_points": [{"x": "any", "y": "any", "series": "string|null"}],
            "trend_summary": "string|null",
            "notes": "string|null",
        }
        instruction = (
            "You extract chart or figure information from an image.\n"
            "Return strict JSON only.\n"
            "Required fields: chart_type, axes, series, data_points, trend_summary, notes.\n"
            f"Schema example: {json.dumps(json_schema, ensure_ascii=False)}\n"
            "If uncertain, use null or empty arrays. Do not include extra text."
        )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        temperature=0,
    )
    text = completion.choices[0].message.content or ""

    try:
        return _extract_json_from_text(text)
    except Exception as e:  # pragma: no cover
        raise VLMError(f"Failed to parse VLM JSON: {e}") from e
