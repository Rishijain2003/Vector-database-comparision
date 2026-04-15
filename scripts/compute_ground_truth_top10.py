#!/usr/bin/env python3
"""
Phase 2 — Brute-force cosine top-k for each query in query_slice.npz vs corpus Parquet.

Requires benchmarks/query_slice.npz from materialize_query_slice.py.

Writes benchmarks/ground_truth_top10.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ingest_common import DEFAULT_PARQUET, EMB_COL  # noqa: E402

EXPECTED_CORPUS_ROWS = 10_000


def _rel_to_repo(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(p.resolve())


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def _l2_normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, eps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=Path(os.environ.get("PARQUET_PATH", DEFAULT_PARQUET)))
    parser.add_argument("--queries-npz", type=Path, default=BENCHMARKS_DIR / "query_slice.npz")
    parser.add_argument("--manifest", type=Path, default=BENCHMARKS_DIR / "query_slice_manifest.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=BENCHMARKS_DIR / "ground_truth_top10.json")
    args = parser.parse_args()

    if args.k < 1:
        raise SystemExit("--k must be >= 1")

    parquet = args.parquet.resolve()
    npz_path = args.queries_npz.resolve()
    if not parquet.is_file():
        raise SystemExit(f"Corpus Parquet not found: {parquet}")
    if not npz_path.is_file():
        raise SystemExit(f"Query npz not found: {npz_path}\nRun: python scripts/materialize_query_slice.py")

    manifest: dict | None = None
    man_path = args.manifest.resolve()
    if man_path.is_file():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))

    print("Loading corpus embeddings from Parquet …")
    df = pd.read_parquet(parquet)
    if EMB_COL not in df.columns:
        raise SystemExit(f"Missing column {EMB_COL!r}")
    if len(df) != EXPECTED_CORPUS_ROWS:
        print(
            f"Warning: corpus has {len(df)} rows, expected {EXPECTED_CORPUS_ROWS}. "
            "Brute force uses all loaded rows as corpus.",
        )

    corpus_ids = df["_id"].astype(str).to_numpy()
    corpus_titles = df["title"].astype(str).to_numpy()
    X = np.stack(df[EMB_COL].to_numpy()).astype(np.float32, copy=False)
    Xn = _l2_normalize_rows(X)
    n_corpus = X.shape[0]

    npz = np.load(npz_path, allow_pickle=True)
    Q = npz["vectors"].astype(np.float32)
    q_ids = [str(x) for x in np.asarray(npz["query_ids"]).tolist()]
    offsets = np.asarray(npz["global_offsets"], dtype=np.int64)
    if "query_titles" in npz.files:
        q_titles = [str(x) for x in np.asarray(npz["query_titles"]).tolist()]
    else:
        q_titles = [""] * len(q_ids)
    if "query_texts" in npz.files:
        q_texts = [str(x) for x in np.asarray(npz["query_texts"]).tolist()]
    else:
        q_texts = [""] * len(q_ids)

    num_queries = Q.shape[0]
    if len(q_ids) != num_queries or len(offsets) != num_queries:
        raise SystemExit("npz vectors/query_ids/global_offsets length mismatch")
    if len(q_titles) < num_queries:
        q_titles.extend([""] * (num_queries - len(q_titles)))
    elif len(q_titles) > num_queries:
        q_titles = q_titles[:num_queries]
    if len(q_texts) < num_queries:
        q_texts.extend([""] * (num_queries - len(q_texts)))
    elif len(q_texts) > num_queries:
        q_texts = q_texts[:num_queries]

    skip = int(offsets[0]) if num_queries else 0
    if num_queries > 1 and not np.all(np.diff(offsets) == 1):
        print("Warning: global_offsets are not consecutive.")

    print(f"Brute-force cosine top-{args.k} for {num_queries} queries vs corpus (n={n_corpus}) …")
    t_bf0 = time.perf_counter()
    results: list[dict] = []
    k = min(args.k, n_corpus)

    for qi in range(num_queries):
        q = Q[qi]
        qn = _l2_normalize_vec(q)
        sims = Xn @ qn
        if k < n_corpus:
            part = np.argpartition(-sims, k - 1)[:k]
            top_local = part[np.argsort(-sims[part])]
        else:
            top_local = np.argsort(-sims)[:k]

        top_list = []
        for rank, corpus_row in enumerate(top_local.tolist(), start=1):
            top_list.append(
                {
                    "rank": rank,
                    "corpus_row": corpus_row,
                    "corpus_id": corpus_ids[corpus_row],
                    "similarity": float(sims[corpus_row]),
                    "corpus_title": corpus_titles[corpus_row],
                }
            )

        go = int(offsets[qi])
        results.append(
            {
                "query_i": qi,
                "global_hf_offset": go,
                "query_id": q_ids[qi],
                "query_title": q_titles[qi],
                "query_text": q_texts[qi],
                "top10": top_list,
            }
        )

    bf_seconds = time.perf_counter() - t_bf0
    print(f"Brute force done in {bf_seconds:.2f}s ({num_queries / max(bf_seconds, 1e-9):.1f} queries/s)")

    dataset_id = (manifest or {}).get("dataset_id", "unknown")
    emb_col_manifest = (manifest or {}).get("embedding_column", EMB_COL)

    payload = {
        "version": 1,
        "metric": "cosine_topk_brute_force",
        "dataset_id": dataset_id,
        "embedding_column": emb_col_manifest,
        "corpus_parquet": str(parquet),
        "corpus_rows": n_corpus,
        "query_slice_npz": _rel_to_repo(npz_path),
        "query_slice_manifest": _rel_to_repo(man_path) if man_path.is_file() else None,
        "query_stream_skip": (manifest or {}).get("query_stream_skip", skip),
        "num_queries": num_queries,
        "k": args.k,
        "brute_force_wall_seconds": round(bf_seconds, 4),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    out_json = args.output_json.resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
