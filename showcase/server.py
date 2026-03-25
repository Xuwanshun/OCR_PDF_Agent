from __future__ import annotations

from argparse import ArgumentParser
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ocr_agent.qa import answer_with_sources


ROOT_DIR = Path(__file__).resolve().parents[1]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    db_dir: str = Field(default="chroma_db")


class SourceItem(BaseModel):
    chunk_id: str
    page: int | None = None
    chunk_type: str | None = None
    source_path: str
    source_name: str
    distance: float
    similarity: float
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    top_k: int
    db_dir: str
    sources: list[SourceItem]


def _to_sources(chunks) -> list[SourceItem]:
    out: list[SourceItem] = []
    for c in chunks:
        source_path = str(c.metadata.get("source_path", ""))
        source_name = Path(source_path).name
        preview = (c.text or "").replace("\n", " ").strip()
        if len(preview) > 420:
            preview = preview[:420] + "..."
        out.append(
            SourceItem(
                chunk_id=c.chunk_id,
                page=c.metadata.get("page"),
                chunk_type=c.metadata.get("chunk_type"),
                source_path=source_path,
                source_name=source_name,
                distance=float(c.distance),
                similarity=float(1.0 - c.distance),
                preview=preview,
            )
        )
    return out


def _resolve_db_dir(raw: str) -> Path:
    db_dir = Path(raw.strip() or "chroma_db")
    if not db_dir.is_absolute():
        db_dir = ROOT_DIR / db_dir
    return db_dir


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_dotenv(override=True)
    yield


app = FastAPI(title="OCR Showcase API", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/showcase/")


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    db_dir = _resolve_db_dir(req.db_dir)
    if not db_dir.exists():
        raise HTTPException(status_code=400, detail=f"db_dir does not exist: {db_dir}")

    try:
        answer, chunks = answer_with_sources(question=req.question, db_dir=db_dir, top_k=req.top_k)
        sources = _to_sources(chunks)
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        question=req.question,
        answer=answer,
        top_k=req.top_k,
        db_dir=str(db_dir),
        sources=sources,
    )


# Mount static content after API routes so /api/* is not shadowed.
app.mount("/", StaticFiles(directory=str(ROOT_DIR), html=True), name="static")


def main() -> None:
    parser = ArgumentParser(description="FastAPI showcase server with live /api/ask endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"Serving showcase at http://{args.host}:{args.port}/showcase/")
    print(f"Live query endpoint: POST http://{args.host}:{args.port}/api/ask")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
