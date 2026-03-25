from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import chromadb


COLLECTION_NAME = "ocr_documents"


def get_collection(db_dir: Path):
    client = chromadb.PersistentClient(path=str(db_dir))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def collection_count(*, db_dir: Path) -> int:
    return int(get_collection(db_dir).count())


def doc_chunk_count(*, db_dir: Path, doc_hash: str) -> int:
    col = get_collection(db_dir)
    res = col.get(where={"doc_hash": doc_hash}, include=[])
    ids = res.get("ids") or []
    if ids and isinstance(ids[0], list):
        # Defensive for unexpected nested shapes.
        ids = ids[0]
    return len(ids)


def _get_openai_client():
    from openai import OpenAI  # type: ignore

    return OpenAI()


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    client = _get_openai_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def add_chunks(
    *,
    db_dir: Path,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]],
):
    if not ids:
        return
    embed_model = os.getenv("OCR_EMBED_MODEL", "text-embedding-3-small")
    embeddings = []
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        embeddings.extend(embed_texts(batch, model=embed_model))

    col = get_collection(db_dir)
    # Use upsert for idempotency across re-runs.
    col.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float


def query(
    *,
    db_dir: Path,
    question: str,
    top_k: int,
) -> list[RetrievedChunk]:
    embed_model = os.getenv("OCR_EMBED_MODEL", "text-embedding-3-small")
    q_emb = embed_texts([question], model=embed_model)[0]

    col = get_collection(db_dir)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]

    out: list[RetrievedChunk] = []
    for cid, txt, meta, dist in zip(ids, docs, metas, dists):
        out.append(RetrievedChunk(chunk_id=cid, text=txt or "", metadata=meta or {}, distance=float(dist)))
    return out
