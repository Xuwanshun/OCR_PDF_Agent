from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import os
import numpy as np
from PIL import Image


class MissingDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class OCRItem:
    text: str
    bbox: list[float]  # x0,y0,x1,y1
    confidence: float


def _poly_to_xyxy(poly: list[list[float]]) -> list[float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def ocr_image_paddle(image_path: str) -> list[OCRItem]:
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as e:  # pragma: no cover
        raise MissingDependencyError(
            "PaddleOCR not installed. Run: `uv sync --extra ocr` or use --ocr-engine vision."
        ) from e

    ocr = PaddleOCR(lang="en", show_log=False)
    result = ocr.ocr(image_path, cls=False) or []

    items: list[OCRItem] = []
    for page in result:
        for entry in page:
            if not entry or len(entry) < 2:
                continue
            poly = entry[0]
            text, score = entry[1][0], entry[1][1]
            if not text or not str(text).strip():
                continue
            items.append(OCRItem(text=str(text).strip(), bbox=_poly_to_xyxy(poly), confidence=float(score)))
    return items


def ocr_image_vision(image_path: str) -> list[OCRItem]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise MissingDependencyError("openai package not available for vision OCR") from e

    import base64

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    model = os.getenv("OCR_VISION_MODEL", "gpt-4o-mini")
    client = OpenAI()
    instruction = (
        "Extract all readable text from this page image. "
        "Return plain text only, preserving line breaks as much as possible. "
        "Do not add explanations."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        temperature=0,
    )
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        return []
    return [OCRItem(text=text, bbox=[0.0, 0.0, float(w), float(h)], confidence=0.5)]
