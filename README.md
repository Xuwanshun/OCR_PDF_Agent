# OCR PDF Agent

An end-to-end PDF intelligence + RAG system that:

- Ingests PDFs from `Document/`
- Extracts text + visual regions (table/chart/figure)
- Writes structured artifacts (`document.json`, `document.md`, `metadata.json`)
- Indexes chunks into local Chroma (`chroma_db/`)
- Answers questions with retrieval-grounded citations
- Includes a live web demo with API-backed query input

## GitHub Pages Demo (Static)

Public static demo:

- `https://xuwanshun.github.io/OCR_PDF_Agent/showcase/`

What it is:

- A hosted static showcase for project overview, pipeline, and evaluation visualization.
- Great for sharing demo pages without running local servers.

Important:

- GitHub Pages is static only, so live FastAPI query (`/api/ask`) is not available there.
- For live querying, run local FastAPI server (`uv run python showcase/server.py`).

## Project Explanation Outline


1. Problem: long PDFs are hard to search, compare, and query reliably.
2. Solution: OCR PDF Agent converts PDFs into structured memory and supports cited QA.
3. Pipeline: ingest -> extract text/visuals -> chunk/index -> retrieve -> answer with sources.
4. Validation: golden question benchmark reports retrieval hit rate, pass rate, and citation quality.
5. Demo: ask a live question in the showcase web app and inspect cited chunks/pages.

## Architecture At A Glance

1. Ingest + extraction: `ocr_agent/ingest.py`
2. OCR engines: `ocr_agent/ocr.py` (`paddle` or `vision`)
3. Visual extraction (VLM): `ocr_agent/vlm.py`
4. Chunking + vector store: `ocr_agent/chunking.py`, `ocr_agent/rag_store.py`
5. QA flow: `ocr_agent/qa.py`
6. CLI: `ocr_agent/cli.py`
7. Live web API (FastAPI): `showcase/server.py`

## Quick Start

### 1) Install

```bash
uv sync
```

### 2) Configure Environment

Copy `.env.example` to `.env` and set:

```bash
OPENAI_API_KEY=...
```

Optional model overrides:

```bash
OCR_CHAT_MODEL=gpt-4o-mini
OCR_VISION_MODEL=gpt-4o-mini
OCR_EMBED_MODEL=text-embedding-3-small
```

### 3) Add PDFs

Place source files under `Document/`.

### 4) Ingest + Index

```bash
uv run ocr ingest --input-dir ./Document --out-dir ./outputs --db ./chroma_db --ocr-engine vision
```

### 5) Ask Questions (CLI)

```bash
uv run ocr ask "Name the four core AI RMF functions." --db ./chroma_db --top-k 6
uv run ocr chat --db ./chroma_db --top-k 6
```

## Command Cookbook

Rebuild index from existing `outputs/*/document.json` artifacts:

```bash
uv run ocr reindex --out-dir ./outputs --db ./chroma_db
```

Estimate ingest/query cost without writing artifacts:

```bash
uv run ocr ingest --cost-estimate-only
```

Run tests:

```bash
uv run pytest
```

## Live Demo Website

Start FastAPI showcase server:

```bash
uv run python showcase/server.py
```

Open:

- Showcase UI: `http://localhost:8765/showcase/`
- API docs: `http://localhost:8765/docs`
- Health check: `http://localhost:8765/healthz`

Live query endpoint:

- `POST /api/ask` with JSON body:

```json
{
  "question": "Name the four core AI RMF functions.",
  "top_k": 6,
  "db_dir": "chroma_db"
}
```

## Evaluation

Run golden-set evaluation:

```bash
uv run python evaluation/golden_eval_vlm.py \
  --db chroma_db \
  --questions evaluation/golden_questions_vlm.json \
  --out-json evaluation/golden_eval_vlm.json \
  --out-md evaluation/golden_eval_vlm.md \
  --use-llm-judge
```

Key outputs:

- `evaluation/golden_eval_vlm.md`
- `evaluation/golden_eval_vlm.json`

## Notes On OCR / VLM

- Default strategy: use native PDF text when available; fallback to OCR for scanned pages.
- Default OCR engine is `paddle`.
- If you do not install PaddleOCR extras, use `--ocr-engine vision`.

Install PaddleOCR extras:

```bash
uv sync --extra ocr
```

## Troubleshooting

If live query always returns "I don't know":

1. Ensure index is populated:
   `uv run ocr reindex --out-dir ./outputs --db ./chroma_db`
2. Verify API key is loaded in `.env`:
   `OPENAI_API_KEY=...`
3. Confirm server is running:
   `http://localhost:8765/healthz`

If port `8765` is already in use:

```bash
lsof -nP -iTCP:8765 -sTCP:LISTEN
kill <PID>
```
