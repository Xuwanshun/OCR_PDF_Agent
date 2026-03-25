from __future__ import annotations

import os
from pathlib import Path

from .rag_store import RetrievedChunk, query


def _get_openai_client():
    from openai import OpenAI  # type: ignore

    return OpenAI()


def _chat_completion(*, prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required (set it in .env).")

    model = os.getenv("OCR_CHAT_MODEL", "gpt-4o-mini")
    client = _get_openai_client()

    system = (
        "You are a careful document QA assistant. "
        "Answer only from the retrieved context. "
        "If the context is insufficient, clearly say you don't know and suggest re-ingesting PDFs or asking a narrower question."
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return (completion.choices[0].message.content or "").strip()


def _format_sources(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for c in chunks:
        sim = 1.0 - c.distance
        page = c.metadata.get("page", "?")
        ctype = c.metadata.get("chunk_type", "?")
        src = c.metadata.get("source_path", "")
        preview = (c.text or "").replace("\n", " ").strip()
        if len(preview) > 420:
            preview = preview[:420] + "..."
        lines.append(f"- p{page} [{ctype}] sim={sim:.3f} {Path(str(src)).name}: {preview}")
    return "\n".join(lines)


def answer_with_sources(*, question: str, db_dir: Path, top_k: int) -> tuple[str, list[RetrievedChunk]]:
    chunks = query(db_dir=db_dir, question=question, top_k=top_k)
    if not chunks:
        return (
            "I could not retrieve any indexed context from the local Chroma DB. "
            "Please run `ocr ingest` (or point to the correct `--db` directory) and try again.",
            chunks,
        )
    context = "\n\n".join(
        [
            f"[source: {Path(str(c.metadata.get('source_path',''))).name} p{c.metadata.get('page')} {c.metadata.get('chunk_type')} id={c.chunk_id}]\n{c.text}"
            for c in chunks
        ]
    )
    prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    answer = _chat_completion(prompt=prompt)
    return answer, chunks


def ask_once(*, question: str, db_dir: Path, top_k: int) -> None:
    answer, chunks = answer_with_sources(question=question, db_dir=db_dir, top_k=top_k)
    print(answer)
    print("\nSources:\n" + _format_sources(chunks))


def chat_loop(*, db_dir: Path, top_k: int) -> None:
    print("OCR chat. Type :q to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q in (":q", "quit", "exit"):
            break
        ask_once(question=q, db_dir=db_dir, top_k=top_k)
