
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from endee.constants import DEFAULT_M, Precision  # noqa: E402
from endee_client import resolve_endee_client  # noqa: E402
from ingest_common import DEFAULT_PARQUET, EMB_COL, clip_filter_for_endee  # noqa: E402

INDEX_NAME = "ef_con_trial"
EF_CON = 100
METRICS_FILE = REPO_ROOT / "metrics_indexing_ef_con.json"
METRICS_KEY = "ef_con_trial"


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


def _merge_metrics(path: Path, key: str, payload: dict) -> None:
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[key] = payload
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=Path(os.environ.get("PARQUET_PATH", DEFAULT_PARQUET)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("INGEST_BATCH_SIZE", "1000")))
    parser.add_argument(
        "--keep-index",
        action="store_true",
        help="Do not delete ef_con_trial if present; create only when missing, then upsert",
    )
    args = parser.parse_args()

    if args.batch_size > 1000:
        raise SystemExit("Endee allows at most 1000 vectors per upsert batch")

    parquet = args.parquet.resolve()
    if not parquet.is_file():
        raise SystemExit(f"Parquet not found: {parquet}")

    client, api_base, auth_mode = resolve_endee_client()
    name = INDEX_NAME

    existing = _index_names(client)
    if name in existing and not args.keep_index:
        print(f"Removing existing trial index {name!r} so it can be created with ef_con={EF_CON} …")
        client.delete_index(name)
        existing = _index_names(client)

    if name not in existing:
        print(f"Creating index {name!r} (dim=1536, cosine, M={DEFAULT_M}, ef_con={EF_CON}, precision=FLOAT32) …")
        client.create_index(
            name=name,
            dimension=1536,
            space_type="cosine",
            M=DEFAULT_M,
            ef_con=EF_CON,
            precision=Precision.FLOAT32,
        )

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
        "backend": "endee_native_ef_con_trial",
        "script": "ingest_endee_ef_con_trial.py",
        "index_name": name,
        "ef_con": EF_CON,
        "M": index.M,
        "ef_con_server": index.ef_con,
        "precision": str(index.precision),
        "indexing_seconds": round(elapsed, 4),
        "num_rows": n,
        "parquet": str(parquet),
        "endee_api_base": api_base,
        "endee_auth_mode": auth_mode,
        "batch_size": args.batch_size,
        "vector_dimension": 1536,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _merge_metrics(METRICS_FILE, METRICS_KEY, payload)
    print(f"Done in {elapsed:.2f}s — wrote metrics to {METRICS_FILE} under {METRICS_KEY!r}")


if __name__ == "__main__":
    main()
