from __future__ import annotations

import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from .agent_graph import ask_once_agent, chat_loop_agent
from .ingest_graph import ingest_directory_agent
from .ingest import ingest_directory, reindex_outputs
from .qa import ask_once, chat_loop

app = typer.Typer(add_completion=False, help="OCR PDF Agent (PDF→JSON/Markdown + RAG + QA)")


@app.command()
def ingest(
    input_dir: Path = typer.Option(Path("./Document"), exists=True, file_okay=False, dir_okay=True),
    out_dir: Path = typer.Option(Path("./outputs"), file_okay=False, dir_okay=True),
    db: Path = typer.Option(Path("./chroma_db"), file_okay=False, dir_okay=True),
    ocr_engine: str = typer.Option("paddle", help="OCR engine for scanned pages: paddle|vision"),
    strict: bool = typer.Option(False, help="If true: no PDFs found => exit code 1."),
    cost_estimate_only: bool = typer.Option(
        False,
        "--cost-estimate-only",
        help="Estimate ingest/query cost from current PDFs and exit without ingesting.",
    ),
):
    """
    Ingest PDFs under INPUT_DIR into JSON/Markdown outputs + Chroma RAG memory.
    """
    load_dotenv(override=True)
    ingest_directory(
        input_dir=input_dir,
        out_dir=out_dir,
        db_dir=db,
        ocr_engine=ocr_engine,
        strict=strict,
        cost_estimate_only=cost_estimate_only,
    )


@app.command("ingest-agent")
def ingest_agent(
    input_dir: Path = typer.Option(Path("./Document"), exists=True, file_okay=False, dir_okay=True),
    out_dir: Path = typer.Option(Path("./outputs"), file_okay=False, dir_okay=True),
    db: Path = typer.Option(Path("./chroma_db"), file_okay=False, dir_okay=True),
    ocr_engine: str = typer.Option("paddle", help="OCR engine for scanned pages: paddle|vision"),
    strict: bool = typer.Option(False, help="If true: no PDFs found => exit code 1."),
    cost_estimate_only: bool = typer.Option(
        False,
        "--cost-estimate-only",
        help="Estimate ingest/query cost from current PDFs and exit without ingesting.",
    ),
):
    """
    Ingest using a LangGraph-orchestrated pipeline (document/page extraction + indexing).
    """
    load_dotenv(override=True)
    ingest_directory_agent(
        input_dir=input_dir,
        out_dir=out_dir,
        db_dir=db,
        ocr_engine=ocr_engine,
        strict=strict,
        cost_estimate_only=cost_estimate_only,
    )


@app.command()
def ask(
    question: str = typer.Argument(...),
    db: Path = typer.Option(Path("./chroma_db"), file_okay=False, dir_okay=True),
    top_k: int = typer.Option(6),
):
    """
    Ask a single question using RAG retrieval from the local Chroma DB.
    """
    load_dotenv(override=True)
    ask_once(question=question, db_dir=db, top_k=top_k)


@app.command()
def chat(
    db: Path = typer.Option(Path("./chroma_db"), file_okay=False, dir_okay=True),
    top_k: int = typer.Option(6),
):
    """
    Interactive chat loop using RAG retrieval from the local Chroma DB.
    """
    load_dotenv(override=True)
    chat_loop(db_dir=db, top_k=top_k)


@app.command("ask-agent")
def ask_agent(
    question: str = typer.Argument(...),
    db: Path = typer.Option(Path("./chroma_db"), file_okay=False, dir_okay=True),
    top_k: int = typer.Option(6),
):
    """
    Ask a single question using LangChain tool-using agent graph + local RAG.
    """
    load_dotenv(override=True)
    ask_once_agent(question=question, db_dir=db, top_k=top_k)


@app.command("chat-agent")
def chat_agent(
    db: Path = typer.Option(Path("./chroma_db"), file_okay=False, dir_okay=True),
    top_k: int = typer.Option(6),
):
    """
    Interactive chat using LangChain tool-using agent graph + local RAG.
    """
    load_dotenv(override=True)
    chat_loop_agent(db_dir=db, top_k=top_k)


@app.command()
def reindex(
    out_dir: Path = typer.Option(Path("./outputs"), file_okay=False, dir_okay=True),
    db: Path = typer.Option(Path("./chroma_db"), file_okay=False, dir_okay=True),
    strict: bool = typer.Option(False, help="If true: no document artifacts found => exit code 1."),
):
    """
    Rebuild/refresh Chroma index from existing outputs/*/document.json artifacts.
    """
    load_dotenv(override=True)
    reindex_outputs(out_dir=out_dir, db_dir=db, strict=strict)


if __name__ == "__main__":
    app()
