from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .chunking import chunk_text
from .costs import build_cost_report, estimate_costs, scan_input_dir
from .markdown import render_markdown
from .ocr import MissingDependencyError, ocr_image_paddle, ocr_image_vision
from .pdf import (
    detect_image_regions,
    detect_table_like_regions_from_text,
    extract_text_blocks,
    iter_pages,
    page_has_enough_text,
    render_page_image,
)
from .rag_store import add_chunks, doc_chunk_count
from .schemas import DocumentJSON, IngestMetadata, PageContent, Region, TextBlock
from .utils import ensure_dir, sha256_file
from .vlm import extract_region, image_file_to_base64_png


def _sort_blocks_reading_order(blocks: list[TextBlock]) -> list[TextBlock]:
    return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))


def _pdf_bbox_to_image_bbox(bbox_pdf: list[float], *, scale_x: float, scale_y: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_pdf
    return (
        int(x0 * scale_x),
        int(y0 * scale_y),
        int(x1 * scale_x),
        int(y1 * scale_y),
    )


def _image_bbox_to_pdf_bbox(bbox_img: list[float], *, scale_x: float, scale_y: float) -> list[float]:
    x0, y0, x1, y1 = bbox_img
    return [float(x0 / scale_x), float(y0 / scale_y), float(x1 / scale_x), float(y1 / scale_y)]


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _extract_region_with_retry(*, image_base64: str, region_type: str, retries: int = 6) -> dict[str, Any]:
    delay = 0.5
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return extract_region(image_base64=image_base64, region_type=region_type)
        except Exception as e:
            last_error = e
            if attempt == retries - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unknown VLM extraction error")


def _build_chunks_from_doc_payload(
    *,
    doc_hash: str,
    source_path: str,
    pages: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    chunk_metas: list[dict[str, Any]] = []

    for page in pages:
        page_number = int(page.get("page_number") or 0)
        if page_number <= 0:
            continue

        # Text chunks
        blocks = page.get("blocks") or []
        page_text = "\n\n".join(
            (str(b.get("text", "")).strip())
            for b in blocks
            if str(b.get("text", "")).strip()
        )
        for i, c in enumerate(chunk_text(page_text)):
            cid = f"{doc_hash}:p{page_number}:text:{i}"
            chunk_ids.append(cid)
            chunk_texts.append(c)
            chunk_metas.append(
                {
                    "doc_hash": doc_hash,
                    "source_path": source_path,
                    "page": page_number,
                    "chunk_type": "text",
                }
            )

        # Region chunks
        for region in page.get("regions") or []:
            extraction = region.get("extraction")
            if not extraction:
                continue
            region_type = region.get("type")
            if region_type == "table":
                chunk_type = "table"
            elif region_type in ("chart", "figure"):
                chunk_type = "chart"
            else:
                continue

            region_id = int(region.get("region_id") or 0)
            bbox = region.get("bbox") or [None, None, None, None]
            if len(bbox) < 4:
                bbox = [None, None, None, None]

            cid = f"{doc_hash}:p{page_number}:{chunk_type}:{region_id}"
            chunk_ids.append(cid)
            chunk_texts.append(json.dumps(extraction, ensure_ascii=False))
            chunk_metas.append(
                {
                    "doc_hash": doc_hash,
                    "source_path": source_path,
                    "page": page_number,
                    "chunk_type": chunk_type,
                    "region_id": region_id,
                    "bbox_x0": bbox[0],
                    "bbox_y0": bbox[1],
                    "bbox_x1": bbox[2],
                    "bbox_y1": bbox[3],
                }
            )

    return chunk_ids, chunk_texts, chunk_metas


def _reindex_existing_doc_if_needed(
    *,
    pdf_path: Path,
    doc_hash: str,
    doc_out: Path,
    db_dir: Path,
) -> bool:
    indexed_count = doc_chunk_count(db_dir=db_dir, doc_hash=doc_hash)
    if indexed_count > 0:
        print(f"[skip] {pdf_path.name} ({doc_hash}) already indexed ({indexed_count} chunks)")
        return True

    doc_json_path = doc_out / "document.json"
    if not doc_json_path.exists():
        print(f"[warn] {pdf_path.name} has metadata but no document.json; cannot reindex automatically")
        return True

    data = json.loads(doc_json_path.read_text(encoding="utf-8"))
    source_path = str(data.get("source_path") or pdf_path)
    pages = data.get("pages") or []
    chunk_ids, chunk_texts, chunk_metas = _build_chunks_from_doc_payload(
        doc_hash=doc_hash,
        source_path=source_path,
        pages=pages,
    )
    if not chunk_ids:
        print(f"[warn] {pdf_path.name} has no chunks to reindex")
        return True

    add_chunks(db_dir=db_dir, ids=chunk_ids, texts=chunk_texts, metadatas=chunk_metas)
    print(f"[reindex] {pdf_path.name} ({doc_hash}) restored {len(chunk_ids)} chunks")
    return True


def ingest_directory(
    *,
    input_dir: Path,
    out_dir: Path,
    db_dir: Path,
    ocr_engine: str,
    strict: bool,
    cost_estimate_only: bool = False,
):
    pdfs = sorted([p for p in input_dir.rglob("*.pdf") if p.is_file()])
    if not pdfs:
        msg = f"No PDFs found under {input_dir}"
        if strict:
            raise SystemExit(msg)
        print(msg)
        return

    if cost_estimate_only:
        embed_model = os.getenv("OCR_EMBED_MODEL", "text-embedding-3-small")
        vision_model = os.getenv("OCR_VISION_MODEL", "gpt-4o-mini")
        qa_model = os.getenv("OCR_CHAT_MODEL", "gpt-4o-mini")
        stats = scan_input_dir(input_dir)
        estimate = estimate_costs(
            stats=stats,
            embed_model=embed_model,
            vision_model=vision_model,
            qa_model=qa_model,
        )
        print(
            build_cost_report(
                stats=stats,
                estimate=estimate,
                embed_model=embed_model,
                vision_model=vision_model,
                qa_model=qa_model,
                vlm_input_tokens_per_region=1200,
                vlm_output_tokens_per_region=250,
                qa_input_tokens=2500,
                qa_output_tokens=400,
            )
        )
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for ingest (embeddings/VLM). Set it in .env.")

    ensure_dir(out_dir)
    ensure_dir(db_dir)

    for pdf_path in pdfs:
        doc_hash = sha256_file(pdf_path)
        doc_out = out_dir / doc_hash
        ensure_dir(doc_out)
        ensure_dir(doc_out / "regions")
        ensure_dir(doc_out / "pages")

        meta_path = doc_out / "metadata.json"
        if meta_path.exists():
            # Incremental mode: ensure pre-existing artifacts are also indexed in Chroma.
            _reindex_existing_doc_if_needed(
                pdf_path=pdf_path,
                doc_hash=doc_hash,
                doc_out=doc_out,
                db_dir=db_dir,
            )
            continue

        pages: list[PageContent] = []

        region_id_counter = 1

        for page_idx, page in enumerate(iter_pages(str(pdf_path))):
            page_number = page_idx + 1
            width, height = float(page.rect.width), float(page.rect.height)

            pdf_blocks = extract_text_blocks(page)
            page_img: Image.Image = render_page_image(page, dpi=200)
            scale_x = page_img.width / width if width else 1.0
            scale_y = page_img.height / height if height else 1.0
            page_img_path = doc_out / "pages" / f"page_{page_number}.png"
            page_img.save(page_img_path)

            blocks: list[TextBlock] = []
            if page_has_enough_text(pdf_blocks):
                for b in pdf_blocks:
                    blocks.append(TextBlock(text=b.text, bbox=b.bbox, confidence=1.0, source="pdf"))
            else:
                if ocr_engine == "paddle":
                    try:
                        ocr_items = ocr_image_paddle(str(page_img_path))
                    except MissingDependencyError as e:
                        raise SystemExit(str(e))
                    for it in ocr_items:
                        bbox_pdf = _image_bbox_to_pdf_bbox(it.bbox, scale_x=scale_x, scale_y=scale_y)
                        blocks.append(TextBlock(text=it.text, bbox=bbox_pdf, confidence=it.confidence, source="ocr"))
                elif ocr_engine == "vision":
                    ocr_items = ocr_image_vision(str(page_img_path))
                    for it in ocr_items:
                        bbox_pdf = _image_bbox_to_pdf_bbox(it.bbox, scale_x=scale_x, scale_y=scale_y)
                        blocks.append(TextBlock(text=it.text, bbox=bbox_pdf, confidence=it.confidence, source="ocr"))
                else:
                    raise SystemExit("Invalid --ocr-engine. Use: paddle|vision")

            blocks = _sort_blocks_reading_order(blocks)

            page_content = PageContent(page_number=page_number, width=width, height=height, blocks=blocks, regions=[])

            # Region detection (best-effort): image regions + table-like text blocks
            regions: list[Region] = []
            for r in detect_image_regions(page):
                rid = region_id_counter
                region_id_counter += 1
                crop_box = _pdf_bbox_to_image_bbox(r.bbox, scale_x=scale_x, scale_y=scale_y)
                crop = page_img.crop(crop_box)
                region_img_path = doc_out / "regions" / f"page_{page_number}_{rid}.png"
                crop.save(region_img_path)
                regions.append(
                    Region(region_id=rid, type="figure", page=page_number, bbox=r.bbox, image_path=str(region_img_path))
                )

            for bbox in detect_table_like_regions_from_text(pdf_blocks):
                rid = region_id_counter
                region_id_counter += 1
                crop_box = _pdf_bbox_to_image_bbox(bbox, scale_x=scale_x, scale_y=scale_y)
                crop = page_img.crop(crop_box)
                region_img_path = doc_out / "regions" / f"page_{page_number}_{rid}.png"
                crop.save(region_img_path)
                regions.append(
                    Region(region_id=rid, type="table", page=page_number, bbox=bbox, image_path=str(region_img_path))
                )

            if regions:
                for region in regions:
                    if region.type not in ("table", "chart", "figure"):
                        continue
                    b64 = image_file_to_base64_png(region.image_path or "")
                    try:
                        region.extraction = _extract_region_with_retry(
                            image_base64=b64,
                            region_type="table" if region.type == "table" else "chart",
                        )
                    except Exception as e:
                        print(
                            f"[warn] VLM extraction failed for {pdf_path.name} page={page_number} "
                            f"region={region.region_id} type={region.type}: {e}"
                        )
                        region.extraction = None

            page_content.regions = regions
            pages.append(page_content)

        doc_json = DocumentJSON(doc_hash=doc_hash, source_path=str(pdf_path), pages=pages)
        _save_json(doc_out / "document.json", doc_json.model_dump())
        (doc_out / "document.md").write_text(render_markdown(doc_json), encoding="utf-8")

        meta = IngestMetadata(
            doc_hash=doc_hash,
            source_path=str(pdf_path),
            pages=len(pages),
            vlm_enabled=True,
            ocr_engine=ocr_engine,
        )
        _save_json(meta_path, meta.model_dump())

        # Persist to Chroma
        chunk_ids, chunk_texts, chunk_metas = _build_chunks_from_doc_payload(
            doc_hash=doc_hash,
            source_path=str(pdf_path),
            pages=[p.model_dump() for p in pages],
        )
        add_chunks(db_dir=db_dir, ids=chunk_ids, texts=chunk_texts, metadatas=chunk_metas)
        print(f"[ok] {pdf_path.name} -> {doc_out}")


def reindex_outputs(
    *,
    out_dir: Path,
    db_dir: Path,
    strict: bool = False,
) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for reindex (embeddings). Set it in .env.")

    ensure_dir(db_dir)
    doc_json_paths = sorted(p for p in out_dir.glob("*/document.json") if p.is_file())
    if not doc_json_paths:
        msg = f"No document.json artifacts found under {out_dir}"
        if strict:
            raise SystemExit(msg)
        print(msg)
        return

    for doc_json_path in doc_json_paths:
        data = json.loads(doc_json_path.read_text(encoding="utf-8"))
        doc_hash = str(data.get("doc_hash") or doc_json_path.parent.name)
        source_path = str(data.get("source_path") or "")
        pages = data.get("pages") or []
        chunk_ids, chunk_texts, chunk_metas = _build_chunks_from_doc_payload(
            doc_hash=doc_hash,
            source_path=source_path,
            pages=pages,
        )
        if not chunk_ids:
            print(f"[warn] {doc_json_path} produced no chunks")
            continue
        add_chunks(db_dir=db_dir, ids=chunk_ids, texts=chunk_texts, metadatas=chunk_metas)
        source_name = Path(source_path).name if source_path else doc_hash
        print(f"[reindex] {source_name} -> {len(chunk_ids)} chunks")
