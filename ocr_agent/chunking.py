from __future__ import annotations

from typing import Iterable


def chunk_text(text: str, *, target_chars: int = 4800, max_chars: int = 8000) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0

    def flush():
        nonlocal buf, size
        if buf:
            chunks.append("\n\n".join(buf).strip())
        buf = []
        size = 0

    for p in parts:
        if len(p) > max_chars:
            flush()
            for i in range(0, len(p), max_chars):
                chunks.append(p[i : i + max_chars])
            continue

        if size + len(p) > target_chars and buf:
            flush()

        buf.append(p)
        size += len(p)

    flush()
    return [c for c in chunks if c.strip()]

