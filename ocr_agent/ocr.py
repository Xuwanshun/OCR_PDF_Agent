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


def _poly_to_xyxy_safe(poly: Any, *, fallback: list[float]) -> list[float]:
    try:
        arr = np.asarray(poly, dtype=float)
        if arr.size < 8:
            return fallback
        arr = arr.reshape(-1, 2)
        xs = arr[:, 0].tolist()
        ys = arr[:, 1].tolist()
        return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    except Exception:
        return fallback


def _result_obj_to_dict(v: Any) -> dict[str, Any] | None:
    if isinstance(v, dict):
        return v
    to_dict = getattr(v, "to_dict", None)
    if callable(to_dict):
        try:
            out = to_dict()
            if isinstance(out, dict):
                return out
        except Exception:
            pass
    if hasattr(v, "__dict__"):
        try:
            out = vars(v)
            if isinstance(out, dict):
                return out
        except Exception:
            pass
    return None


def _extract_items_from_page_result(page: Any, *, w: int, h: int) -> list[OCRItem]:
    items: list[OCRItem] = []
    fallback_bbox = [0.0, 0.0, float(w), float(h)]

    # Legacy PaddleOCR shape:
    # [[poly, (text, score)], ...]
    if isinstance(page, list):
        for entry in page:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                poly = entry[0]
                rec = entry[1]
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    text = str(rec[0]).strip()
                    if not text:
                        continue
                    try:
                        score = float(rec[1])
                    except Exception:
                        score = 0.5
                    items.append(
                        OCRItem(
                            text=text,
                            bbox=_poly_to_xyxy_safe(poly, fallback=fallback_bbox),
                            confidence=score,
                        )
                    )
                    continue

            d = _result_obj_to_dict(entry)
            if not d:
                continue
            texts = d.get("rec_texts") or d.get("texts") or []
            scores = d.get("rec_scores") or d.get("scores") or []
            polys = d.get("rec_polys") or d.get("dt_polys") or d.get("polys") or []
            for i, t in enumerate(texts):
                text = str(t).strip()
                if not text:
                    continue
                try:
                    score = float(scores[i]) if i < len(scores) else 0.5
                except Exception:
                    score = 0.5
                poly = polys[i] if i < len(polys) else None
                items.append(
                    OCRItem(
                        text=text,
                        bbox=_poly_to_xyxy_safe(poly, fallback=fallback_bbox),
                        confidence=score,
                    )
                )
        return items

    # PaddleOCR v3 result object/dict shape:
    # {"rec_texts": [...], "rec_scores": [...], "rec_polys"/"dt_polys": [...]}
    d = _result_obj_to_dict(page)
    if not d:
        return items
    texts = d.get("rec_texts") or d.get("texts") or []
    scores = d.get("rec_scores") or d.get("scores") or []
    polys = d.get("rec_polys") or d.get("dt_polys") or d.get("polys") or []
    for i, t in enumerate(texts):
        text = str(t).strip()
        if not text:
            continue
        try:
            score = float(scores[i]) if i < len(scores) else 0.5
        except Exception:
            score = 0.5
        poly = polys[i] if i < len(polys) else None
        items.append(
            OCRItem(
                text=text,
                bbox=_poly_to_xyxy_safe(poly, fallback=fallback_bbox),
                confidence=score,
            )
        )
    return items


def ocr_image_paddle(image_path: str) -> list[OCRItem]:
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as e:  # pragma: no cover
        raise MissingDependencyError(
            "PaddleOCR not installed. Run: `uv sync --extra ocr` or use --ocr-engine vision."
        ) from e

    # PaddleOCR v3 removed/changed some constructor args (e.g. show_log).
    # Try the older signature first, then fall back to a minimal compatible call.
    try:
        ocr = PaddleOCR(lang="en", show_log=False)
    except (TypeError, ValueError):
        ocr = PaddleOCR(lang="en")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # API compatibility across PaddleOCR versions.
    try:
        result = ocr.ocr(image_path, cls=False)
    except TypeError:
        try:
            result = ocr.ocr(image_path)
        except TypeError:
            result = ocr.predict(image_path)

    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        pages = list(result)
    else:
        try:
            pages = list(result)
        except TypeError:
            pages = [result]

    items: list[OCRItem] = []
    for page in pages:
        items.extend(_extract_items_from_page_result(page, w=w, h=h))
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
