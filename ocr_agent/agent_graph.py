from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .qa import _format_sources
from .rag_store import RetrievedChunk, query


def _require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required (set it in .env).")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict):
                txt = part.get("text")
                if isinstance(txt, str) and txt.strip():
                    out.append(txt.strip())
            elif isinstance(part, str) and part.strip():
                out.append(part.strip())
        return "\n".join(out).strip()
    return str(content or "").strip()


def _tool_result_from_chunks(*, chunks: list[RetrievedChunk], heading: str) -> str:
    if not chunks:
        return f"{heading}\nNo matching chunks found."

    lines: list[str] = [heading]
    for c in chunks:
        page = c.metadata.get("page", "?")
        ctype = c.metadata.get("chunk_type", "?")
        src = Path(str(c.metadata.get("source_path", ""))).name
        sim = 1.0 - c.distance
        preview = (c.text or "").replace("\n", " ").strip()
        if len(preview) > 600:
            preview = preview[:600] + "..."
        lines.append(f"- [source: {src} p{page} {ctype} id={c.chunk_id}] sim={sim:.3f}")
        lines.append(preview)
    return "\n".join(lines)


def _build_agent(*, default_db_dir: Path, default_top_k: int):
    # Lazy imports so base pipeline works without agent extras installed.
    try:
        from langchain_core.tools import tool  # type: ignore
        from langchain_openai import ChatOpenAI  # type: ignore
        from langgraph.prebuilt import create_react_agent  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "LangChain agent dependencies are missing. Run: `uv sync --extra agent`."
        ) from e

    @tool
    def retrieve_context(question: str, top_k: int = default_top_k, db_dir: str = str(default_db_dir)) -> str:
        """
        Retrieve relevant chunks from local Chroma RAG memory for a question.
        Use this first before answering.
        """
        chunks = query(db_dir=Path(db_dir), question=question, top_k=max(1, int(top_k)))
        return _tool_result_from_chunks(chunks=chunks, heading="Retrieved context:")

    @tool
    def analyze_table(question: str, top_k: int = default_top_k, db_dir: str = str(default_db_dir)) -> str:
        """
        Retrieve and summarize table-oriented chunks from local memory.
        Use when the user asks for tabular values, rows, columns, or comparisons.
        """
        chunks = query(db_dir=Path(db_dir), question=question, top_k=max(1, int(top_k) * 2))
        table_chunks = [c for c in chunks if str(c.metadata.get("chunk_type")) == "table"][: max(1, int(top_k))]
        return _tool_result_from_chunks(chunks=table_chunks, heading="Table analysis context:")

    @tool
    def analyze_chart(question: str, top_k: int = default_top_k, db_dir: str = str(default_db_dir)) -> str:
        """
        Retrieve and summarize chart/figure-oriented chunks from local memory.
        Use when the user asks about trends, axes, chart type, or plotted values.
        """
        chunks = query(db_dir=Path(db_dir), question=question, top_k=max(1, int(top_k) * 2))
        chart_chunks = [c for c in chunks if str(c.metadata.get("chunk_type")) == "chart"][: max(1, int(top_k))]
        return _tool_result_from_chunks(chunks=chart_chunks, heading="Chart analysis context:")

    model = os.getenv("OCR_CHAT_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    prompt = (
        "You are an OCR+RAG agent. "
        "Use tools to retrieve evidence before answering. "
        "Cite sources with [source: <doc.pdf> p<page> <type> id=<chunk_id>] when possible. "
        "If context is insufficient, say you don't know."
    )
    return create_react_agent(llm, tools=[retrieve_context, analyze_table, analyze_chart], prompt=prompt)


def answer_with_agent_graph(*, question: str, db_dir: Path, top_k: int) -> tuple[str, list[RetrievedChunk]]:
    _require_openai_api_key()
    agent = _build_agent(default_db_dir=db_dir, default_top_k=top_k)

    try:
        from langchain_core.messages import HumanMessage, AIMessage  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "LangChain agent dependencies are missing. Run: `uv sync --extra agent`."
        ) from e

    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = result.get("messages") or []

    answer = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            # Final assistant output usually has no tool calls.
            if not getattr(m, "tool_calls", None):
                answer = _content_to_text(m.content)
                if answer:
                    break
    if not answer:
        answer = "I don't know based on the available context."

    # Keep source output format aligned with existing CLI behavior.
    chunks = query(db_dir=db_dir, question=question, top_k=top_k)
    return answer, chunks


def ask_once_agent(*, question: str, db_dir: Path, top_k: int) -> None:
    answer, chunks = answer_with_agent_graph(question=question, db_dir=db_dir, top_k=top_k)
    print(answer)
    print("\nSources:\n" + _format_sources(chunks))


def chat_loop_agent(*, db_dir: Path, top_k: int) -> None:
    print("OCR agent chat (LangChain tool graph). Type :q to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q in (":q", "quit", "exit"):
            break
        ask_once_agent(question=q, db_dir=db_dir, top_k=top_k)
