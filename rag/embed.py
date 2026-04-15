"""Embed user queries with OpenAI ``text-embedding-3-large`` at 1536 dims (matches indexed corpus)."""

from __future__ import annotations

import os

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1536


def embed_query(text: str) -> list[float]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set OPENAI_API_KEY in your environment or repo .env")

    from openai import OpenAI

    client = OpenAI(api_key=key)
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return list(resp.data[0].embedding)
