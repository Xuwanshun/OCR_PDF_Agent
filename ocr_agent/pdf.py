from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, Optional

import fitz  # PyMuPDF
from PIL import Image


@dataclass(frozen=True)
class PDFBlock:
    text: str
    bbox: list[float]  # x0,y0,x1,y1


@dataclass(frozen=True)
class PDFImageRegion:
    bbox: list[float]


def iter_pages(pdf_path: str) -> Iterable[fitz.Page]:
    doc = fitz.open(pdf_path)
    try:
        for i in range(doc.page_count):
            yield doc.load_page(i)
    finally:
        doc.close()


def extract_text_blocks(page: fitz.Page) -> list[PDFBlock]:
    blocks_raw = page.get_text("blocks") or []
    blocks: list[PDFBlock] = []
    for b in blocks_raw:
        if len(b) < 5:
            continue
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        if not text or not str(text).strip():
            continue
        blocks.append(PDFBlock(text=str(text).strip(), bbox=[float(x0), float(y0), float(x1), float(y1)]))
    return blocks


def page_has_enough_text(blocks: list[PDFBlock], min_chars: int = 40) -> bool:
    return sum(len(b.text) for b in blocks) >= min_chars


def render_page_image(page: fitz.Page, dpi: int = 200) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    png_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def detect_image_regions(page: fitz.Page) -> list[PDFImageRegion]:
    regions: list[PDFImageRegion] = []
    for img in page.get_images(full=True) or []:
        xref = img[0]
        rects = page.get_image_rects(xref)
        for r in rects:
            regions.append(PDFImageRegion(bbox=[float(r.x0), float(r.y0), float(r.x1), float(r.y1)]))
    return regions


def detect_table_like_regions_from_text(blocks: list[PDFBlock]) -> list[list[float]]:
    def looks_like_table(text: str) -> bool:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 3:
            return False
        multi_col_lines = 0
        for ln in lines:
            if "|" in ln:
                multi_col_lines += 1
                continue
            if "  " in ln or "\t" in ln:
                multi_col_lines += 1
        return multi_col_lines >= max(2, len(lines) // 2)

    table_regions: list[list[float]] = []
    for b in blocks:
        if looks_like_table(b.text):
            table_regions.append(b.bbox)
    return table_regions

