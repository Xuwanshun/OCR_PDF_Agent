from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, TypedDict

from PIL import Image

from .ingest import (
    _build_chunks_from_doc_payload,
    _extract_region_with_retry,
    _image_bbox_to_pdf_bbox,
    _pdf_bbox_to_image_bbox,
    _reindex_existing_doc_if_needed,
    _save_json,
    _sort_blocks_reading_order,
    ingest_directory,
)
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
from .rag_store import add_chunks
from .schemas import DocumentJSON, IngestMetadata, PageContent, Region, TextBlock
from .utils import ensure_dir, sha256_file
from .vlm import image_file_to_base64_png


class IngestGraphState(TypedDict, total=False):
    input_dir: Path
    out_dir: Path
    db_dir: Path
    ocr_engine: Literal["paddle", "vision"]
    strict: bool
    pdfs: list[Path]
    pdf_index: int
    done: bool
    current_pdf: Path
    current_doc_hash: str
    current_doc_out: Path
    current_meta_path: Path
    skip_current: bool
    pages: list[PageContent]
    chunk_count: int


def _build_ingest_graph():
    try:
        from langgraph.graph import END, START, StateGraph  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "LangGraph dependencies are missing. Run: `uv sync --extra agent`."
        ) from e

    def discover_pdfs(state: IngestGraphState) -> IngestGraphState:
        input_dir = Path(state["input_dir"])
        strict = bool(state.get("strict", False))
        pdfs = sorted([p for p in input_dir.rglob("*.pdf") if p.is_file()])
        if not pdfs:
            msg = f"No PDFs found under {input_dir}"
            if strict:
                raise SystemExit(msg)
            print(msg)
            return {"pdfs": [], "pdf_index": 0, "done": True}
        return {"pdfs": pdfs, "pdf_index": 0, "done": False}

    def init_dirs(state: IngestGraphState) -> IngestGraphState:
        ensure_dir(Path(state["out_dir"]))
        ensure_dir(Path(state["db_dir"]))
        return {}

    def select_pdf(state: IngestGraphState) -> IngestGraphState:
        pdfs = state.get("pdfs", [])
        idx = int(state.get("pdf_index", 0))
        if idx >= len(pdfs):
            return {"done": True}

        pdf_path = pdfs[idx]
        doc_hash = sha256_file(pdf_path)
        doc_out = Path(state["out_dir"]) / doc_hash
        ensure_dir(doc_out)
        ensure_dir(doc_out / "regions")
        ensure_dir(doc_out / "pages")
        return {
            "done": False,
            "current_pdf": pdf_path,
            "current_doc_hash": doc_hash,
            "current_doc_out": doc_out,
            "current_meta_path": doc_out / "metadata.json",
            "pages": [],
            "chunk_count": 0,
        }

    def route_select(state: IngestGraphState) -> str:
        return "finish" if state.get("done", False) else "maybe_skip_existing"

    def maybe_skip_existing(state: IngestGraphState) -> IngestGraphState:
        pdf_path = Path(state["current_pdf"])
        doc_hash = str(state["current_doc_hash"])
        doc_out = Path(state["current_doc_out"])
        db_dir = Path(state["db_dir"])
        meta_path = Path(state["current_meta_path"])

        if meta_path.exists():
            _reindex_existing_doc_if_needed(
                pdf_path=pdf_path,
                doc_hash=doc_hash,
                doc_out=doc_out,
                db_dir=db_dir,
            )
            return {"skip_current": True}
        return {"skip_current": False}

    def route_skip(state: IngestGraphState) -> str:
        return "advance_pdf" if state.get("skip_current", False) else "extract_page_content"

    def extract_page_content(state: IngestGraphState) -> IngestGraphState:
        pdf_path = Path(state["current_pdf"])
        doc_out = Path(state["current_doc_out"])
        ocr_engine = str(state["ocr_engine"])

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

            ordered_blocks = _sort_blocks_reading_order(blocks)
            page_content = PageContent(
                page_number=page_number,
                width=width,
                height=height,
                blocks=ordered_blocks,
                regions=[],
            )

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

        return {"pages": pages}

    def write_outputs(state: IngestGraphState) -> IngestGraphState:
        pdf_path = Path(state["current_pdf"])
        doc_hash = str(state["current_doc_hash"])
        doc_out = Path(state["current_doc_out"])
        ocr_engine = str(state["ocr_engine"])
        pages = state.get("pages", [])

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
        _save_json(Path(state["current_meta_path"]), meta.model_dump())
        return {}

    def index_doc_chunks(state: IngestGraphState) -> IngestGraphState:
        pdf_path = Path(state["current_pdf"])
        doc_hash = str(state["current_doc_hash"])
        db_dir = Path(state["db_dir"])
        pages = state.get("pages", [])

        chunk_ids, chunk_texts, chunk_metas = _build_chunks_from_doc_payload(
            doc_hash=doc_hash,
            source_path=str(pdf_path),
            pages=[p.model_dump() for p in pages],
        )
        add_chunks(db_dir=db_dir, ids=chunk_ids, texts=chunk_texts, metadatas=chunk_metas)
        print(f"[ok][agent] {pdf_path.name} -> {Path(state['current_doc_out'])} ({len(chunk_ids)} chunks)")
        return {"chunk_count": len(chunk_ids)}

    def advance_pdf(state: IngestGraphState) -> IngestGraphState:
        idx = int(state.get("pdf_index", 0))
        return {"pdf_index": idx + 1, "skip_current": False, "pages": []}

    graph = StateGraph(IngestGraphState)
    graph.add_node("discover_pdfs", discover_pdfs)
    graph.add_node("init_dirs", init_dirs)
    graph.add_node("select_pdf", select_pdf)
    graph.add_node("maybe_skip_existing", maybe_skip_existing)
    graph.add_node("extract_page_content", extract_page_content)
    graph.add_node("write_outputs", write_outputs)
    graph.add_node("index_doc_chunks", index_doc_chunks)
    graph.add_node("advance_pdf", advance_pdf)

    graph.add_edge(START, "discover_pdfs")
    graph.add_edge("discover_pdfs", "init_dirs")
    graph.add_edge("init_dirs", "select_pdf")
    graph.add_conditional_edges(
        "select_pdf",
        route_select,
        {"maybe_skip_existing": "maybe_skip_existing", "finish": END},
    )
    graph.add_conditional_edges(
        "maybe_skip_existing",
        route_skip,
        {"advance_pdf": "advance_pdf", "extract_page_content": "extract_page_content"},
    )
    graph.add_edge("extract_page_content", "write_outputs")
    graph.add_edge("write_outputs", "index_doc_chunks")
    graph.add_edge("index_doc_chunks", "advance_pdf")
    graph.add_edge("advance_pdf", "select_pdf")

    return graph.compile()


def ingest_directory_agent(
    *,
    input_dir: Path,
    out_dir: Path,
    db_dir: Path,
    ocr_engine: str,
    strict: bool,
    cost_estimate_only: bool = False,
) -> None:
    if cost_estimate_only:
        ingest_directory(
            input_dir=input_dir,
            out_dir=out_dir,
            db_dir=db_dir,
            ocr_engine=ocr_engine,
            strict=strict,
            cost_estimate_only=True,
        )
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for ingest (embeddings/VLM). Set it in .env.")

    app = _build_ingest_graph()
    app.invoke(
        {
            "input_dir": input_dir,
            "out_dir": out_dir,
            "db_dir": db_dir,
            "ocr_engine": ocr_engine,
            "strict": strict,
        }
    )
