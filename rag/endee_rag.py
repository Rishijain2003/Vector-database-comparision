#!/usr/bin/env python3
"""
RAG CLI: embed query → retrieve from Endee → answer with LLM (prompt in ``llm_prompt.txt``).

Defaults match your ingest: index ``dbpedia_10k_benchmark`` (override with ENDEE_INDEX_NAME).

Run from repo root:
  python rag/endee_rag.py "Who is Albert Einstein?"

Prints retrieved chunks (with similarity), then the LLM answer on stdout.

Env: OPENAI_API_KEY, OPENAI_CHAT_MODEL (optional), ENDEE_* via ``scripts/endee_client.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_RAG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _RAG_DIR.parent
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from embed import embed_query  # noqa: E402
from endee_client import resolve_endee_client  # noqa: E402
from env_loader import load_repo_dotenv  # noqa: E402
from llm import answer_with_context  # noqa: E402


def _format_endee_hits(rows: list, max_chars: int = 12000) -> str:
    parts: list[str] = []
    for i, item in enumerate(rows, start=1):
        meta = item.get("meta") or {}
        title = str(meta.get("title", "")).strip() or "(no title)"
        body = str(meta.get("text", "")).strip()
        rid = str(meta.get("_id", "")).strip()
        sim = item.get("similarity")
        score_line = f"similarity: {sim}\n" if sim is not None else ""
        parts.append(f"[{i}] id={rid}\n{score_line}title: {title}\n{body}\n")
    text = "\n".join(parts)
    if len(text) > max_chars:
        return text[:max_chars] + "\n…(truncated)"
    return text


def main() -> None:
    load_repo_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="User question")
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("RAG_TOP_K", "5")))
    parser.add_argument("--ef", type=int, default=int(os.environ.get("ENDEE_EF", "128")))
    parser.add_argument(
        "--index-name",
        default=os.environ.get("ENDEE_INDEX_NAME", "dbpedia_10k_benchmark"),
    )
    args = parser.parse_args()

    client, _, _ = resolve_endee_client()
    index = client.get_index(name=args.index_name)
    qvec = embed_query(args.question)
    rows = index.query(
        vector=qvec,
        top_k=args.top_k,
        ef=args.ef,
        include_vectors=False,
    )
    context = _format_endee_hits(rows)
    print("\n=== Retrieved chunks (Endee) ===\n")
    print(context)
    print("\n=== LLM answer ===\n")
    answer = answer_with_context(question=args.question, context=context)
    print(answer)


if __name__ == "__main__":
    main()
