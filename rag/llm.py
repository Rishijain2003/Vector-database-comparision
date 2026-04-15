"""Call chat LLM with RAG system prompt + retrieved context."""

from __future__ import annotations

import os
from pathlib import Path

_RAG_DIR = Path(__file__).resolve().parent


def load_system_prompt() -> str:
    path = _RAG_DIR / "llm_prompt.txt"
    if not path.is_file():
        raise SystemExit(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()


def answer_with_context(*, question: str, context: str) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set OPENAI_API_KEY in your environment or repo .env")

    model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
    system = load_system_prompt()
    user = (
        "CONTEXT (excerpts from retrieved documents):\n"
        "-----\n"
        f"{context}\n"
        "-----\n\n"
        f"QUESTION:\n{question}\n"
    )

    from openai import OpenAI

    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=float(os.environ.get("OPENAI_CHAT_TEMPERATURE", "0.2")),
    )
    choice = resp.choices[0].message.content
    return (choice or "").strip()
