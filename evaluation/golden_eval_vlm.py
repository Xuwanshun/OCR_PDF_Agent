from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ocr_agent.qa import _chat_completion
from ocr_agent.rag_store import query


UNKNOWN_MARKERS = (
    "i don't know",
    "i do not know",
    "cannot find",
    "not available",
    "unable to",
    "insufficient context",
)


@dataclass
class CitationStats:
    citations_found: int
    citations_valid: int
    citation_valid_rate: float


@dataclass
class EvalItemResult:
    id: str
    category: str
    question: str
    answer: str
    retrieved_docs: list[str]
    retrieved_types: list[str]
    expected_doc_hit: bool
    expected_type_hit: bool
    required_terms: list[str]
    required_terms_matched: list[str]
    term_recall: float
    citation_stats: CitationStats
    abstained: bool
    deterministic_pass: bool
    llm_judge: dict[str, Any] | None


def _safe_rate(num: int, den: int) -> float:
    return round((num / den), 3) if den else 0.0


def _term_recall(answer: str, required_terms: list[str]) -> tuple[list[str], float]:
    if not required_terms:
        return [], 1.0
    lower = answer.lower()
    matched = [t for t in required_terms if t.lower() in lower]
    return matched, _safe_rate(len(matched), len(required_terms))


def _build_context_and_sources(db_dir: Path, question: str, top_k: int) -> tuple[str, list[dict[str, Any]]]:
    chunks = query(db_dir=db_dir, question=question, top_k=top_k)
    source_rows: list[dict[str, Any]] = []
    parts: list[str] = []
    for c in chunks:
        doc = Path(str(c.metadata.get("source_path", ""))).name
        page = c.metadata.get("page")
        ctype = str(c.metadata.get("chunk_type", "unknown"))
        parts.append(f"[source: {doc} p{page} {ctype} id={c.chunk_id}]\n{c.text}")
        source_rows.append(
            {
                "doc": doc,
                "page": int(page) if isinstance(page, (int, float)) else None,
                "chunk_type": ctype,
                "similarity": round(1.0 - c.distance, 4),
            }
        )
    return "\n\n".join(parts), source_rows


def _citation_stats(answer: str, source_rows: list[dict[str, Any]]) -> CitationStats:
    # Accept both:
    # 1) [doc.pdf p12]
    # 2) [source: doc.pdf p12 ...]
    citation_pairs: list[tuple[str, int]] = []
    for m in re.finditer(r"\[([^\[\]]+?\.pdf)\s+p(\d+)\]", answer, flags=re.IGNORECASE):
        citation_pairs.append((Path(m.group(1).strip()).name, int(m.group(2))))
    for m in re.finditer(r"\[source:\s*([^\[\]]+?\.pdf)\s+p(\d+)(?:\s+[^\]]*)?\]", answer, flags=re.IGNORECASE):
        citation_pairs.append((Path(m.group(1).strip()).name, int(m.group(2))))

    if not citation_pairs:
        return CitationStats(citations_found=0, citations_valid=0, citation_valid_rate=0.0)

    valid = 0
    valid_set = {(s["doc"], s["page"]) for s in source_rows if s["page"] is not None}
    for doc, page in citation_pairs:
        if (doc, page) in valid_set:
            valid += 1
    return CitationStats(
        citations_found=len(citation_pairs),
        citations_valid=valid,
        citation_valid_rate=_safe_rate(valid, len(citation_pairs)),
    )


def _build_answer_prompt(question: str, context: str) -> str:
    return (
        "You are evaluating a RAG system answer quality.\n"
        "Answer in English only.\n"
        "Use ONLY the provided context.\n"
        "For every factual claim, cite at least one source.\n"
        "Preferred format: [doc.pdf p12].\n"
        "Also acceptable: [source: doc.pdf p12 type id=...].\n"
        "If information is missing, say \"I don't know\".\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def _judge_with_llm(
    *,
    question: str,
    answer: str,
    expected_docs: list[str],
    expected_types: list[str],
    required_terms: list[str],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from openai import OpenAI  # type: ignore

    model = os.getenv("OCR_JUDGE_MODEL", "gpt-4o-mini")
    client = OpenAI()
    prompt = (
        "You are a strict evaluator. Return JSON only.\n"
        "Score from 0 to 1.\n"
        "Fields: correctness, citation_support, overall, notes.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Expected docs: {expected_docs}\n"
        f"Expected chunk types: {expected_types}\n"
        f"Required terms: {required_terms}\n"
        f"Retrieved sources: {json.dumps(source_rows, ensure_ascii=False)}\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    return {"correctness": 0.0, "citation_support": 0.0, "overall": 0.0, "notes": "judge_parse_failed"}


def evaluate_one(
    *,
    item: dict[str, Any],
    db_dir: Path,
    top_k: int,
    use_llm_judge: bool,
) -> EvalItemResult:
    qid = str(item["id"])
    category = str(item.get("category", "general"))
    question = str(item["question"])
    expected_docs = [str(x) for x in item.get("expected_docs", [])]
    expected_types = [str(x) for x in item.get("expected_chunk_types", [])]
    required_terms = [str(x) for x in item.get("required_terms", [])]
    min_term_recall = float(item.get("min_term_recall", 0.5))

    context, source_rows = _build_context_and_sources(db_dir=db_dir, question=question, top_k=top_k)
    answer = _chat_completion(prompt=_build_answer_prompt(question, context))

    retrieved_docs = [str(s["doc"]) for s in source_rows]
    retrieved_types = [str(s["chunk_type"]) for s in source_rows]
    expected_doc_hit = (not expected_docs) or any(d in expected_docs for d in retrieved_docs)
    expected_type_hit = (not expected_types) or any(t in expected_types for t in retrieved_types)

    matched, term_recall = _term_recall(answer, required_terms)
    cstats = _citation_stats(answer, source_rows)
    lower_answer = answer.lower()
    abstained = any(marker in lower_answer for marker in UNKNOWN_MARKERS)

    deterministic_pass = (
        expected_doc_hit
        and expected_type_hit
        and term_recall >= min_term_recall
        and cstats.citations_found >= 1
        and cstats.citation_valid_rate >= 0.5
        and not abstained
    )

    llm_judge = None
    if use_llm_judge:
        llm_judge = _judge_with_llm(
            question=question,
            answer=answer,
            expected_docs=expected_docs,
            expected_types=expected_types,
            required_terms=required_terms,
            source_rows=source_rows,
        )

    return EvalItemResult(
        id=qid,
        category=category,
        question=question,
        answer=answer,
        retrieved_docs=retrieved_docs,
        retrieved_types=retrieved_types,
        expected_doc_hit=expected_doc_hit,
        expected_type_hit=expected_type_hit,
        required_terms=required_terms,
        required_terms_matched=matched,
        term_recall=term_recall,
        citation_stats=cstats,
        abstained=abstained,
        deterministic_pass=deterministic_pass,
        llm_judge=llm_judge,
    )


def build_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines: list[str] = []
    lines.append("# Golden QA Evaluation (Full OCR System)")
    lines.append("")
    lines.append("## Method")
    lines.append("- Fixed golden question set with expected docs/types and required terms.")
    lines.append("- Deterministic scoring on retrieval hit, term recall, citation validity, and abstention.")
    lines.append("- Optional LLM-as-judge score is reported as secondary evidence.")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Questions: {s['total_questions']}")
    lines.append(f"- Deterministic pass: {s['deterministic_pass_count']}/{s['total_questions']} ({s['deterministic_pass_rate']})")
    lines.append(f"- Expected doc hit: {s['expected_doc_hit_count']}/{s['total_questions']} ({s['expected_doc_hit_rate']})")
    lines.append(f"- Expected type hit: {s['expected_type_hit_count']}/{s['total_questions']} ({s['expected_type_hit_rate']})")
    lines.append(f"- Mean term recall: {s['mean_term_recall']}")
    lines.append(f"- Citation valid rate (mean): {s['mean_citation_valid_rate']}")
    lines.append(f"- Abstain count: {s['abstain_count']}/{s['total_questions']}")
    if "llm_judge_mean_overall" in s:
        lines.append(f"- LLM judge mean overall: {s['llm_judge_mean_overall']}")
    lines.append("")
    lines.append("## Per Question")
    for item in report["results"]:
        lines.append(
            f"- {item['id']} ({item['category']}): pass={item['deterministic_pass']} "
            f"doc_hit={item['expected_doc_hit']} type_hit={item['expected_type_hit']} "
            f"term_recall={item['term_recall']} citation_valid={item['citation_stats']['citation_valid_rate']} "
            f"abstain={item['abstained']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden QA evaluation for VLM index.")
    parser.add_argument("--db", default="chroma_db", help="Chroma DB directory")
    parser.add_argument("--questions", default="evaluation/golden_questions_vlm.json", help="Golden questions JSON file")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--out-json", default="evaluation/golden_eval_vlm.json")
    parser.add_argument("--out-md", default="evaluation/golden_eval_vlm.md")
    parser.add_argument("--use-llm-judge", action="store_true")
    args = parser.parse_args()

    load_dotenv(override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for evaluation.")

    root = Path(__file__).resolve().parent.parent
    db_dir = (root / args.db).resolve()
    questions_path = (root / args.questions).resolve()
    out_json = (root / args.out_json).resolve()
    out_md = (root / args.out_md).resolve()

    items = json.loads(questions_path.read_text(encoding="utf-8"))
    results: list[EvalItemResult] = []
    for item in items:
        results.append(
            evaluate_one(
                item=item,
                db_dir=db_dir,
                top_k=args.top_k,
                use_llm_judge=args.use_llm_judge,
            )
        )

    total = len(results)
    det_pass = sum(1 for r in results if r.deterministic_pass)
    doc_hit = sum(1 for r in results if r.expected_doc_hit)
    type_hit = sum(1 for r in results if r.expected_type_hit)
    abstain_count = sum(1 for r in results if r.abstained)
    mean_term_recall = round(sum(r.term_recall for r in results) / total, 3) if total else 0.0
    mean_citation_valid_rate = (
        round(sum(r.citation_stats.citation_valid_rate for r in results) / total, 3) if total else 0.0
    )

    summary: dict[str, Any] = {
        "total_questions": total,
        "deterministic_pass_count": det_pass,
        "deterministic_pass_rate": _safe_rate(det_pass, total),
        "expected_doc_hit_count": doc_hit,
        "expected_doc_hit_rate": _safe_rate(doc_hit, total),
        "expected_type_hit_count": type_hit,
        "expected_type_hit_rate": _safe_rate(type_hit, total),
        "mean_term_recall": mean_term_recall,
        "mean_citation_valid_rate": mean_citation_valid_rate,
        "abstain_count": abstain_count,
    }

    if args.use_llm_judge:
        judge_items = [r.llm_judge for r in results if r.llm_judge is not None]
        if judge_items:
            mean_overall = round(
                sum(float(x.get("overall", 0.0)) for x in judge_items) / len(judge_items), 3
            )
            summary["llm_judge_mean_overall"] = mean_overall

    report = {
        "mode": "ocr_system_golden_eval",
        "db_dir": str(db_dir),
        "questions_file": str(questions_path),
        "top_k": args.top_k,
        "use_llm_judge": args.use_llm_judge,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
