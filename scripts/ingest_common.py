"""Shared constants and small helpers for ingest / benchmark scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

EMB_COL = "text-embedding-3-large-1536-embedding"
DEFAULT_PARQUET = Path(__file__).resolve().parent.parent / "data" / "dbpedia_openai3_large_10k.parquet"
METRICS_FILE = Path(__file__).resolve().parent.parent / "metrics_indexing.json"

ENDEE_FILTER_STRING_MAX_BYTES = 1000


def truncate_utf8(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore")


def clip_filter_for_endee(meta: dict[str, object], max_bytes: int = ENDEE_FILTER_STRING_MAX_BYTES) -> dict[str, object]:
    """Shrink string values for Endee `filter` (per-value UTF-8 byte cap)."""
    out: dict[str, object] = {}
    for k, v in meta.items():
        if isinstance(v, str):
            out[k] = truncate_utf8(v, max_bytes)
        else:
            out[k] = v
    return out


def load_id_to_fulltext(
    parquet_path: Path,
    id_column: str = "_id",
    text_column: str = "text",
) -> dict[str, str]:
    df = pd.read_parquet(parquet_path, columns=[id_column, text_column])
    return dict(zip(df[id_column].astype(str), df[text_column].astype(str)))


def merge_indexing_metrics(repo_root: Path, key: str, payload: dict) -> None:
    path = repo_root / "metrics_indexing.json"
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[key] = payload
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
