
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ingest_common import DEFAULT_PARQUET, EMB_COL, merge_indexing_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=Path(os.environ.get("PARQUET_PATH", DEFAULT_PARQUET)))
    parser.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "dbpedia_10k_benchmark_native_qdrant"),
    )
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("INGEST_BATCH_SIZE", "1000")))
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete collection if it exists, then create empty",
    )
    args = parser.parse_args()

    parquet = args.parquet.resolve()
    if not parquet.is_file():
        raise SystemExit(f"Parquet not found: {parquet}")

    url = os.environ.get("QDRANT_URL", "http://192.168.108.123:6333").rstrip("/")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    strict = os.environ.get("QDRANT_STRICT_VERSION_CHECK", "").lower() in ("1", "true", "yes")
    client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False, check_compatibility=strict)

    coll = args.collection
    if args.recreate and client.collection_exists(collection_name=coll):
        client.delete_collection(collection_name=coll)

    if not client.collection_exists(collection_name=coll):
        client.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    title_index = os.environ.get("QDRANT_TITLE_INDEX", "1").lower() not in ("0", "false", "no")
    if title_index:
        try:
            client.create_payload_index(
                collection_name=coll,
                field_name="title",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as e:  # noqa: BLE001 — index may already exist
            print(f"Payload index on title (non-fatal): {e}")

    df = pd.read_parquet(parquet)
    if EMB_COL not in df.columns:
        raise SystemExit(f"Missing embedding column {EMB_COL!r}")
    n = len(df)
    print(f"Upserting {n} points into {coll!r} …")

    t0 = time.perf_counter()
    batch: list[PointStruct] = []
    for i, row in df.iterrows():
        rid = str(row["_id"])
        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, rid))
        vec = row[EMB_COL]
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        body = str(row.get("text", ""))
        batch.append(
            PointStruct(
                id=pid,
                vector=[float(x) for x in vec],
                payload={"_id": rid, "title": str(row.get("title", "")), "text": body},
            )
        )
        if len(batch) >= args.batch_size:
            client.upsert(collection_name=coll, points=batch)
            batch = []
    if batch:
        client.upsert(collection_name=coll, points=batch)

    elapsed = time.perf_counter() - t0
    payload = {
        "backend": "qdrant_native",
        "ingest": "native_qdrant_client",
        "script": "ingest_qdrant.py",
        "indexing_seconds": round(elapsed, 4),
        "num_rows": n,
        "collection_name": coll,
        "qdrant_url": url,
        "batch_size": args.batch_size,
        "vector_dimension": 1536,
        "title_payload_index": title_index,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    merge_indexing_metrics(REPO_ROOT, "qdrant_native", payload)
    print(f"Done in {elapsed:.2f}s — metrics merged into metrics_indexing.json")


if __name__ == "__main__":
    main()
