#!/usr/bin/env python3
"""
Ingest a Parquet corpus into Endee using only the `endee` SDK (no LangChain).

Each point: stable UUID5(_id) as vector id, full `text` + `title` + `_id` in `meta`,
and a small `filter` (truncated `title` + `_id`) for Endee's per-value size limits.

Env:
  ENDEE_BASE_URL, ENDEE_API_KEY / NDD_AUTH_TOKEN / ENDEE_LOCAL_NO_AUTH — see endee_client.py
  ENDEE_INDEX_NAME    default dbpedia_10k_benchmark
  PARQUET_PATH        default data/dbpedia_openai3_large_10k.parquet
  INGEST_BATCH_SIZE   default 1000 (Endee max vectors per upsert batch)
  ENDEE_RECREATE      set to "1" to delete existing index name before create

Writes metrics under key `endee_native` in metrics_indexing.json (merged).

Docs: https://docs.endee.io/overview — indexes, upsert, metadata / filters.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from endee_client import resolve_endee_client  # noqa: E402
from ingest_common import DEFAULT_PARQUET, EMB_COL, clip_filter_for_endee, merge_indexing_metrics  # noqa: E402


def _index_names(client) -> set[str]:
    raw = client.list_indexes()
    names: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, dict):
            n = item.get("index_name") or item.get("name")
            if n:
                names.add(str(n))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=Path(os.environ.get("PARQUET_PATH", DEFAULT_PARQUET)))
    parser.add_argument("--index-name", default=os.environ.get("ENDEE_INDEX_NAME", "dbpedia_10k_benchmark"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("INGEST_BATCH_SIZE", "1000")))
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete index if it exists (same as ENDEE_RECREATE=1)",
    )
    args = parser.parse_args()

    if args.batch_size > 1000:
        raise SystemExit("Endee allows at most 1000 vectors per upsert batch; lower --batch-size")

    parquet = args.parquet.resolve()
    if not parquet.is_file():
        raise SystemExit(f"Parquet not found: {parquet}")

    client, api_base, auth_mode = resolve_endee_client()
    name = args.index_name
    recreate = args.recreate or os.environ.get("ENDEE_RECREATE", "").lower() in ("1", "true", "yes")

    existing = _index_names(client)
    if recreate and name in existing:
        print(f"Deleting existing index {name!r} …")
        client.delete_index(name)

    if name not in _index_names(client):
        print(f"Creating index {name!r} (dim=1536, cosine) …")
        client.create_index(name=name, dimension=1536, space_type="cosine")

    index = client.get_index(name=name)
    df = pd.read_parquet(parquet)
    if EMB_COL not in df.columns:
        raise SystemExit(f"Missing embedding column {EMB_COL!r}")
    n = len(df)
    print(f"Upserting {n} vectors into {name!r} …")

    t0 = time.perf_counter()
    batch: list[dict] = []
    for _, row in df.iterrows():
        rid = str(row["_id"])
        vid = str(uuid.uuid5(uuid.NAMESPACE_DNS, rid))
        vec = row[EMB_COL]
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        title = str(row.get("title", ""))
        body = str(row.get("text", ""))
        meta = {"_id": rid, "title": title, "text": body}
        filt = clip_filter_for_endee({"_id": rid, "title": title})
        batch.append({"id": vid, "vector": [float(x) for x in vec], "meta": meta, "filter": filt})
        if len(batch) >= args.batch_size:
            index.upsert(batch)
            batch = []
    if batch:
        index.upsert(batch)

    elapsed = time.perf_counter() - t0
    payload = {
        "backend": "endee_native",
        "ingest": "native_endee_sdk",
        "script": "ingest_endee.py",
        "indexing_seconds": round(elapsed, 4),
        "num_rows": n,
        "index_name": name,
        "endee_api_base": api_base,
        "endee_auth_mode": auth_mode,
        "batch_size": args.batch_size,
        "vector_dimension": 1536,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    merge_indexing_metrics(REPO_ROOT, "endee_native", payload)
    print(f"Done in {elapsed:.2f}s — metrics merged into metrics_indexing.json")


if __name__ == "__main__":
    main()
