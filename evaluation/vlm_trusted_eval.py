from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from ocr_agent.rag_store import query


EXPECTED = Literal["table", "chart"]


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    expected: EXPECTED


@dataclass
class RetrievalResult:
    question: str
    expected: EXPECTED
    top_types: list[str]
    expected_in_top1: bool
    expected_in_top6: bool
    visual_in_top6: bool
    top1_source: str
    top1_page: int | None


CASES: list[RetrievalCase] = [
    RetrievalCase("From extracted chart data, return chart_type and trend_summary.", "chart"),
    RetrievalCase("Identify a chart chunk and provide axes and one series name.", "chart"),
    RetrievalCase("Find a chart extraction related to Federal Reserve inflation.", "chart"),
    RetrievalCase("Find chart extraction in the OSHA manual and summarize notes.", "chart"),
    RetrievalCase("Find a chart chunk from the NASA handbook.", "chart"),
    RetrievalCase("Return two data_points from extracted chart chunks.", "chart"),
    RetrievalCase("From extracted table data, provide headers and one sample row.", "table"),
    RetrievalCase("Find IRS extracted table headers.", "table"),
    RetrievalCase("Find NASA extracted table with appendix/title headers.", "table"),
    RetrievalCase("Find table extraction that includes a units field.", "table"),
    RetrievalCase("Locate a table chunk and provide row count.", "table"),
    RetrievalCase("Identify a table chunk source page and headers.", "table"),
]


def _nonnull(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def load_vlm_extractions(outputs_dir: Path) -> dict[str, Any]:
    docs = 0
    pages = 0
    regions_total = 0
    extracted_total = 0

    table_total = 0
    table_headers_nonempty = 0
    table_rows_nonempty = 0
    table_units_nonempty = 0

    chart_total = 0
    chart_axes_nonempty = 0
    chart_series_nonempty = 0
    chart_datapoints_nonempty = 0
    chart_trend_nonempty = 0

    for doc_json in outputs_dir.glob("*/document.json"):
        docs += 1
        data = json.loads(doc_json.read_text(encoding="utf-8"))
        pages += len(data.get("pages", []))
        for page in data.get("pages", []):
            for region in page.get("regions", []):
                regions_total += 1
                ext = region.get("extraction")
                if not ext:
                    continue
                extracted_total += 1
                rtype = region.get("type")
                if rtype == "table":
                    table_total += 1
                    if _nonnull(ext.get("headers")):
                        table_headers_nonempty += 1
                    if _nonnull(ext.get("rows")):
                        table_rows_nonempty += 1
                    if _nonnull(ext.get("units")):
                        table_units_nonempty += 1
                elif rtype in ("chart", "figure"):
                    chart_total += 1
                    if _nonnull(ext.get("axes")):
                        chart_axes_nonempty += 1
                    if _nonnull(ext.get("series")):
                        chart_series_nonempty += 1
                    if _nonnull(ext.get("data_points")):
                        chart_datapoints_nonempty += 1
                    if _nonnull(ext.get("trend_summary")):
                        chart_trend_nonempty += 1

    def rate(num: int, den: int) -> float:
        return round((num / den), 3) if den else 0.0

    return {
        "docs": docs,
        "pages": pages,
        "regions_total": regions_total,
        "regions_extracted": extracted_total,
        "region_coverage_rate": rate(extracted_total, regions_total),
        "table_total": table_total,
        "table_headers_nonempty": table_headers_nonempty,
        "table_rows_nonempty": table_rows_nonempty,
        "table_units_nonempty": table_units_nonempty,
        "table_headers_rate": rate(table_headers_nonempty, table_total),
        "table_rows_rate": rate(table_rows_nonempty, table_total),
        "table_units_rate": rate(table_units_nonempty, table_total),
        "chart_total": chart_total,
        "chart_axes_nonempty": chart_axes_nonempty,
        "chart_series_nonempty": chart_series_nonempty,
        "chart_datapoints_nonempty": chart_datapoints_nonempty,
        "chart_trend_nonempty": chart_trend_nonempty,
        "chart_axes_rate": rate(chart_axes_nonempty, chart_total),
        "chart_series_rate": rate(chart_series_nonempty, chart_total),
        "chart_datapoints_rate": rate(chart_datapoints_nonempty, chart_total),
        "chart_trend_rate": rate(chart_trend_nonempty, chart_total),
    }


def run_retrieval_eval(db_dir: Path) -> dict[str, Any]:
    results: list[RetrievalResult] = []

    for case in CASES:
        chunks = query(db_dir=db_dir, question=case.question, top_k=6)
        top_types = [str(c.metadata.get("chunk_type", "unknown")) for c in chunks]
        top1 = chunks[0] if chunks else None
        expected_in_top1 = bool(top_types and top_types[0] == case.expected)
        expected_in_top6 = case.expected in top_types
        visual_in_top6 = any(t in ("table", "chart") for t in top_types)

        top1_source = ""
        top1_page: int | None = None
        if top1 is not None:
            top1_source = str(top1.metadata.get("source_path", ""))
            raw_page = top1.metadata.get("page")
            top1_page = int(raw_page) if isinstance(raw_page, (int, float)) else None

        results.append(
            RetrievalResult(
                question=case.question,
                expected=case.expected,
                top_types=top_types,
                expected_in_top1=expected_in_top1,
                expected_in_top6=expected_in_top6,
                visual_in_top6=visual_in_top6,
                top1_source=top1_source,
                top1_page=top1_page,
            )
        )

    total = len(results)
    expected_top1 = sum(1 for r in results if r.expected_in_top1)
    expected_top6 = sum(1 for r in results if r.expected_in_top6)
    visual_top6 = sum(1 for r in results if r.visual_in_top6)

    def rate(num: int, den: int) -> float:
        return round((num / den), 3) if den else 0.0

    return {
        "cases": total,
        "expected_type_top1_count": expected_top1,
        "expected_type_top1_rate": rate(expected_top1, total),
        "expected_type_top6_count": expected_top6,
        "expected_type_top6_rate": rate(expected_top6, total),
        "visual_top6_count": visual_top6,
        "visual_top6_rate": rate(visual_top6, total),
        "results": [asdict(r) for r in results],
    }


def build_markdown(report: dict[str, Any]) -> str:
    extraction = report["extraction_quality"]
    retrieval = report["retrieval_quality"]
    lines: list[str] = []
    lines.append("# OCR System Trusted Evaluation Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Mode: Full OCR system index (`chroma_db`) with visual extraction enabled")
    lines.append("- Dataset: 5 official PDFs in `Document/`")
    lines.append("- Focus: extraction quality + retrieval grounding for visual queries")
    lines.append("")
    lines.append("## Extraction Quality")
    lines.append(f"- Docs / pages: {extraction['docs']} / {extraction['pages']}")
    lines.append(
        f"- Region extraction coverage: {extraction['regions_extracted']}/{extraction['regions_total']} "
        f"({extraction['region_coverage_rate']})"
    )
    lines.append(
        f"- Table headers populated: {extraction['table_headers_nonempty']}/{extraction['table_total']} "
        f"({extraction['table_headers_rate']})"
    )
    lines.append(
        f"- Table rows populated: {extraction['table_rows_nonempty']}/{extraction['table_total']} "
        f"({extraction['table_rows_rate']})"
    )
    lines.append(
        f"- Chart axes populated: {extraction['chart_axes_nonempty']}/{extraction['chart_total']} "
        f"({extraction['chart_axes_rate']})"
    )
    lines.append(
        f"- Chart series populated: {extraction['chart_series_nonempty']}/{extraction['chart_total']} "
        f"({extraction['chart_series_rate']})"
    )
    lines.append(
        f"- Chart trend populated: {extraction['chart_trend_nonempty']}/{extraction['chart_total']} "
        f"({extraction['chart_trend_rate']})"
    )
    lines.append("")
    lines.append("## Retrieval Quality (12 visual-focused cases)")
    lines.append(
        f"- Expected type @ top1: {retrieval['expected_type_top1_count']}/{retrieval['cases']} "
        f"({retrieval['expected_type_top1_rate']})"
    )
    lines.append(
        f"- Expected type @ top6: {retrieval['expected_type_top6_count']}/{retrieval['cases']} "
        f"({retrieval['expected_type_top6_rate']})"
    )
    lines.append(
        f"- Any visual chunk @ top6: {retrieval['visual_top6_count']}/{retrieval['cases']} "
        f"({retrieval['visual_top6_rate']})"
    )
    lines.append("")
    lines.append("## Trust Notes")
    lines.append("- This report uses deterministic checks (schema-population and retrieval typing) rather than LLM-as-judge scoring.")
    lines.append("- Retrieval uses embedding search over your persisted VLM index and reports chunk-type grounding directly.")
    lines.append("- If needed, add a human-labeled QA set for factual exactness scoring (EM/F1) as a next step.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    load_dotenv(override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for this evaluation.")

    root = Path(__file__).resolve().parent.parent
    outputs_dir = root / "outputs"
    db_dir = root / "chroma_db"
    eval_dir = root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    extraction_quality = load_vlm_extractions(outputs_dir)
    retrieval_quality = run_retrieval_eval(db_dir)

    report = {
        "mode": "ocr_system_full",
        "dataset_dir": str(root / "Document"),
        "outputs_dir": str(outputs_dir),
        "db_dir": str(db_dir),
        "extraction_quality": extraction_quality,
        "retrieval_quality": retrieval_quality,
    }

    (eval_dir / "vlm_trusted_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "vlm_trusted_evaluation.md").write_text(
        build_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "cases": retrieval_quality["cases"]}))


if __name__ == "__main__":
    main()
