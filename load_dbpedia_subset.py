#!/usr/bin/env python3
"""
Stream the first N rows from Qdrant's DBpedia + text-embedding-3-large dataset
and save to a local Parquet file for later indexing (Endee, Qdrant, etc.).

Does not download the full 1M rows; only pulls enough remote data to cover N.

Usage:
  export HF_HOME=/path/to/Endee/.hf-cache   # optional; defaults to ./.hf-cache
  python load_dbpedia_subset.py --num-rows 10000
  python load_dbpedia_subset.py -n 10000 -o data/my_10k.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DATASET_ID = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M"
EMBEDDING_COLUMN = "text-embedding-3-large-1536-embedding"


def default_hf_home(repo_root: Path) -> Path:
    return repo_root / ".hf-cache"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-n",
        "--num-rows",
        type=int,
        default=10_000,
        help="Number of rows to save (default: 10000)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=repo_root / "data" / "dbpedia_openai3_large_10k.parquet",
        help="Output Parquet path (default: ./data/dbpedia_openai3_large_10k.parquet)",
    )
    p.add_argument(
        "--dataset",
        default=DATASET_ID,
        help="Hugging Face dataset id",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    if not os.environ.get("HF_HOME"):
        hf = default_hf_home(repo_root)
        hf.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf)

    if args.num_rows < 1:
        raise SystemExit("--num-rows must be >= 1")

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset, load_dataset

    ds_stream = load_dataset(args.dataset, split="train", streaming=True)

    rows: list[dict] = []
    for i, row in enumerate(ds_stream):
        rows.append(dict(row))
        if i + 1 >= args.num_rows:
            break

    if len(rows) < args.num_rows:
        raise SystemExit(
            f"Only collected {len(rows)} rows (requested {args.num_rows}). "
            "Network or dataset issue."
        )

    dataset = Dataset.from_list(rows)
    dataset.to_parquet(str(args.output))

    sample = rows[0]
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"Wrote {len(rows)} rows -> {args.output}")
    print(f"Columns: {list(sample.keys())}")
    print(f"Vector dim: {len(sample[EMBEDDING_COLUMN])}")
    print(f"First title: {sample['title']!r}")


if __name__ == "__main__":
    main()