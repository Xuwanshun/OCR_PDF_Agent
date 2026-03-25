from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from .chunking import chunk_text
from .pdf import detect_image_regions, detect_table_like_regions_from_text, extract_text_blocks

# Per-1M-token pricing (USD). Keep these defaults aligned with current model defaults in this repo.
EMBED_INPUT_PRICE_PER_1M: dict[str, float] = {
    "text-embedding-3-small": 0.02,
}

CHAT_PRICE_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass(frozen=True)
class DirectoryStats:
    pdf_count: int
    total_pages: int
    total_chars: int
    text_chunk_count: int
    table_like_region_count: int
    figure_region_count: int

    @property
    def approx_text_tokens(self) -> int:
        # Rough token estimate for English-heavy docs.
        return int(self.total_chars / 4)

    @property
    def total_vlm_regions(self) -> int:
        return self.table_like_region_count + self.figure_region_count


@dataclass(frozen=True)
class CostEstimate:
    embedding_cost_usd: float
    vlm_cost_usd: float
    total_ingest_cost_usd: float
    qa_per_question_cost_usd: float


def scan_input_dir(input_dir: Path) -> DirectoryStats:
    pdfs = sorted([p for p in input_dir.rglob("*.pdf") if p.is_file()])
    total_pages = 0
    total_chars = 0
    chunk_count = 0
    table_like_count = 0
    figure_count = 0

    for pdf in pdfs:
        doc = fitz.open(pdf)
        try:
            total_pages += doc.page_count
            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                blocks = extract_text_blocks(page)
                page_text = "\n\n".join([b.text for b in blocks if b.text.strip()])
                total_chars += len(page_text)
                chunk_count += len(chunk_text(page_text))
                table_like_count += len(detect_table_like_regions_from_text(blocks))
                figure_count += len(detect_image_regions(page))
        finally:
            doc.close()

    return DirectoryStats(
        pdf_count=len(pdfs),
        total_pages=total_pages,
        total_chars=total_chars,
        text_chunk_count=chunk_count,
        table_like_region_count=table_like_count,
        figure_region_count=figure_count,
    )


def estimate_costs(
    *,
    stats: DirectoryStats,
    embed_model: str,
    vision_model: str,
    vlm_input_tokens_per_region: int = 1200,
    vlm_output_tokens_per_region: int = 250,
    qa_model: str = "gpt-4o-mini",
    qa_input_tokens: int = 2500,
    qa_output_tokens: int = 400,
) -> CostEstimate:
    embed_price = EMBED_INPUT_PRICE_PER_1M.get(embed_model, 0.0)
    embed_tokens = stats.approx_text_tokens
    embedding_cost = (embed_tokens / 1_000_000) * embed_price

    region_count = stats.total_vlm_regions
    vlm_in_tokens = region_count * vlm_input_tokens_per_region
    vlm_out_tokens = region_count * vlm_output_tokens_per_region
    price = CHAT_PRICE_PER_1M.get(vision_model, {"input": 0.0, "output": 0.0})
    vlm_cost = (vlm_in_tokens / 1_000_000) * price["input"] + (vlm_out_tokens / 1_000_000) * price["output"]

    qa_price = CHAT_PRICE_PER_1M.get(qa_model, {"input": 0.0, "output": 0.0})
    qa_cost = (qa_input_tokens / 1_000_000) * qa_price["input"] + (qa_output_tokens / 1_000_000) * qa_price["output"]

    return CostEstimate(
        embedding_cost_usd=embedding_cost,
        vlm_cost_usd=vlm_cost,
        total_ingest_cost_usd=embedding_cost + vlm_cost,
        qa_per_question_cost_usd=qa_cost,
    )


def build_cost_report(
    *,
    stats: DirectoryStats,
    estimate: CostEstimate,
    embed_model: str,
    vision_model: str,
    qa_model: str,
    vlm_input_tokens_per_region: int,
    vlm_output_tokens_per_region: int,
    qa_input_tokens: int,
    qa_output_tokens: int,
) -> str:
    lines: list[str] = []
    lines.append("=== OCR Cost Estimate ===")
    lines.append(f"PDFs: {stats.pdf_count}")
    lines.append(f"Pages: {stats.total_pages}")
    lines.append(f"Chars: {stats.total_chars}")
    lines.append(f"Approx text tokens: {stats.approx_text_tokens}")
    lines.append(f"Estimated text chunks: {stats.text_chunk_count}")
    lines.append(f"Detected table-like regions: {stats.table_like_region_count}")
    lines.append(f"Detected figure regions: {stats.figure_region_count}")
    lines.append(f"Detected total VLM regions: {stats.total_vlm_regions}")
    lines.append("")
    lines.append(f"Embedding model: {embed_model}")
    lines.append(f"Vision model: {vision_model}")
    lines.append(f"QA model: {qa_model}")
    lines.append(
        f"VLM assumption per region (input/output tokens): {vlm_input_tokens_per_region}/{vlm_output_tokens_per_region}"
    )
    lines.append(f"QA assumption per question (input/output tokens): {qa_input_tokens}/{qa_output_tokens}")
    lines.append("")
    lines.append(f"Estimated embedding cost (ingest): ${estimate.embedding_cost_usd:.4f}")
    lines.append(f"Estimated VLM cost (ingest): ${estimate.vlm_cost_usd:.4f}")
    lines.append(f"Estimated total ingest cost: ${estimate.total_ingest_cost_usd:.4f}")
    lines.append(f"Estimated QA cost per question: ${estimate.qa_per_question_cost_usd:.4f}")
    return "\n".join(lines)
