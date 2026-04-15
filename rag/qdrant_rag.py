#!/usr/bin/env python3
"""
RAG CLI: embed query → retrieve from Qdrant → answer with LLM (prompt in ``llm_prompt.txt``).

Defaults: collection ``dbpedia_10k_benchmark_native``, URL from QDRANT_URL.

Run from repo root:
  python rag/qdrant_rag.py "Who is Albert Einstein?"

Prints retrieved chunks (with scores), then the LLM answer on stdout.

Env: OPENAI_API_KEY, OPENAI_CHAT_MODEL (optional), QDRANT_URL, QDRANT_COLLECTION, QDRANT_API_KEY.
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

from embed import embed_query  # noqa: E402
from env_loader import load_repo_dotenv  # noqa: E402
from llm import answer_with_context  # noqa: E402


def _format_qdrant_hits(points: list, max_chars: int = 12000) -> str:
    parts: list[str] = []
    for i, p in enumerate(points, start=1):
        pl = p.payload or {}
        if not isinstance(pl, dict):
            pl = dict(pl) if pl is not None else {}
        title = str(pl.get("title", "")).strip() or "(no title)"
        body = str(pl.get("text", "")).strip()
        rid = str(pl.get("_id", "")).strip()
        sc = getattr(p, "score", None)
        score_line = f"score: {sc}\n" if sc is not None else ""
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
    parser.add_argument("--hnsw-ef", type=int, default=int(os.environ.get("QDRANT_HNSW_EF", "128")))
    parser.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "dbpedia_10k_benchmark_native"),
    )
    args = parser.parse_args()

    from qdrant_client import QdrantClient
    from qdrant_client.models import SearchParams

    url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    strict = os.environ.get("QDRANT_STRICT_VERSION_CHECK", "").lower() in ("1", "true", "yes")
    qclient = QdrantClient(url=url, api_key=api_key, prefer_grpc=False, check_compatibility=strict)

    qvec = embed_query(args.question)
    resp = qclient.query_points(
        collection_name=args.collection,
        query=qvec,
        limit=args.top_k,
        search_params=SearchParams(hnsw_ef=args.hnsw_ef),
        with_payload=True,
    )
    context = _format_qdrant_hits(resp.points)
    print("\n=== Retrieved chunks (Qdrant) ===\n")
    print(context)
    print("\n=== LLM answer ===\n")
    answer = answer_with_context(question=args.question, context=context)
    print(answer)


if __name__ == "__main__":
    main()
