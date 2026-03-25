# Showcase Website

This folder contains a static demo site for presenting OCR PDF Agent results.

## Run (Static Only)

From repo root:

```bash
python3 -m http.server
```

Then open:

- `http://localhost:8000/showcase/`

## Run (With Live Query API)

From repo root:

```bash
uv run python showcase/server.py
```

Then open:

- `http://localhost:8765/showcase/`
- `http://localhost:8765/docs` (FastAPI OpenAPI docs)

This enables `POST /api/ask`, so the Live Query panel can call your real RAG pipeline and return cited sources.

## What it shows

- System pipeline overview
- Golden evaluation metrics from `evaluation/golden_eval_vlm.json`
- Processed document cards using `outputs/*/metadata.json`
- Interactive question board with filters
- Demo playground: choose or edit sample input and view corresponding returned answers
