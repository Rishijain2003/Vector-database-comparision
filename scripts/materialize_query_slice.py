#!/usr/bin/env python3
"""
Phase 1 — Download and save the next N query rows after the corpus slice.

Persists embeddings, _id, global offset, title, and body text (HF column `text`).

Outputs (default benchmarks/):
  query_slice.npz, query_slice_manifest.json

Env: QUERY_SLICE_NUM_QUERIES (default 100), HF_HOME (defaults to repo .hf-cache)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ingest_common import EMB_COL  # noqa: E402

DATASET_ID = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M"
DEFAULT_QUERY_SKIP = 10_000


def _rel_to_repo(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(p.resolve())


def _ensure_hf_home() -> None:
    if not os.environ.get("HF_HOME"):
        hf = REPO_ROOT / ".hf-cache"
        hf.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", type=int, default=DEFAULT_QUERY_SKIP)
    parser.add_argument(
        "--num-queries",
        type=int,
        default=int(os.environ.get("QUERY_SLICE_NUM_QUERIES", "100")),
    )
    parser.add_argument("--output-npz", type=Path, default=BENCHMARKS_DIR / "query_slice.npz")
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=BENCHMARKS_DIR / "query_slice_manifest.json",
    )
    args = parser.parse_args()

    if args.num_queries < 1:
        raise SystemExit("--num-queries must be >= 1")

    out_npz = args.output_npz.resolve()
    out_manifest = args.output_manifest.resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    _ensure_hf_home()
    from datasets import load_dataset

    print(f"Streaming {args.num_queries} rows from HF (skip={args.skip}) …")
    t0 = time.perf_counter()
    stream = load_dataset(DATASET_ID, split="train", streaming=True)
    it = iter(stream)
    rows = list(islice(it, args.skip, args.skip + args.num_queries))
    if len(rows) < args.num_queries:
        raise SystemExit(
            f"Only got {len(rows)} rows (wanted {args.num_queries}). Dataset end or network issue."
        )
    print(f"Slice fetched in {time.perf_counter() - t0:.2f}s")

    if EMB_COL not in rows[0]:
        raise SystemExit(f"Dataset row missing embedding column {EMB_COL!r}")

    text_key = "text" if "text" in rows[0] else None
    if text_key is None:
        sample_keys = list(rows[0].keys())
        print(
            "Warning: no `text` column; query_texts will be empty. "
            f"First row keys: {sample_keys[:20]}{'…' if len(sample_keys) > 20 else ''}",
        )

    Q = np.stack(
        [[float(x) for x in row[EMB_COL]] for row in rows],
        dtype=np.float32,
    )
    q_ids = [str(row["_id"]) for row in rows]
    q_titles = [str(row.get("title", "")) for row in rows]
    q_texts = [str(row.get("text", "")) if text_key else "" for row in rows]
    offsets = np.arange(args.skip, args.skip + args.num_queries, dtype=np.int64)

    np.savez_compressed(
        out_npz,
        vectors=Q,
        query_ids=np.array(q_ids, dtype=object),
        global_offsets=offsets,
        query_titles=np.array(q_titles, dtype=object),
        query_texts=np.array(q_texts, dtype=object),
    )

    manifest = {
        "version": 1,
        "dataset_id": DATASET_ID,
        "embedding_column": EMB_COL,
        "text_column": text_key,
        "query_stream_skip": args.skip,
        "num_queries": args.num_queries,
        "output_npz": _rel_to_repo(out_npz),
        "output_manifest": _rel_to_repo(out_manifest),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out_npz}")
    print(f"Wrote {out_manifest}")


if __name__ == "__main__":
    main()
