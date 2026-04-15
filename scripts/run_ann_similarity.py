#!/usr/bin/env python3
"""
Phase 3 — For each query vector, retrieve top-k neighbors from Endee and/or Qdrant.

Reads: benchmarks/query_slice.npz

Writes:
  benchmarks/endee_similarity.json
  benchmarks/qdrant_similarity.json
  benchmarks/metrics_retrieval.json  — QPS + latency (mean + p50/p95/p99)

If benchmarks/ground_truth_top10.json exists, k is read from it (e.g. top-10).

Env: QDRANT_URL, QDRANT_COLLECTION, QDRANT_API_KEY, ENDEE_INDEX_NAME, ENDEE_BASE_URL,
     ENDEE_API_KEY / NDD_AUTH_TOKEN / ENDEE_LOCAL_NO_AUTH, BENCHMARK_WARMUP,
     QDRANT_HNSW_EF, ENDEE_EF

This script loads the repo root ``.env`` (if present) without overwriting existing env vars.
For local Endee with only ``ENDEE_BASE_URL`` set (no token), it defaults ``ENDEE_LOCAL_NO_AUTH=1``
unless you already set ``ENDEE_LOCAL_NO_AUTH`` or an API token.

Next: scripts/compute_recall.py

Qdrant client 2.x uses ``query_points`` for dense vector search (``search`` was removed); see
https://qdrant.tech/documentation/ — Search / API & SDKs.

Endee search uses the official Python SDK ``index.query``; see https://docs.endee.io/overview
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from endee_client import resolve_endee_client  # noqa: E402


def _load_repo_dotenv() -> None:
    """Populate os.environ from repo ``.env`` when keys are unset (this script only; no extra deps)."""
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        os.environ[key] = val


def _apply_endee_local_defaults() -> None:
    """
    Self-hosted Endee (e.g. http://192.168.108.123:18080) often uses no API token.

    If ENDEE_BASE_URL is set, no token env vars are set, and ENDEE_LOCAL_NO_AUTH was not
    provided, default ENDEE_LOCAL_NO_AUTH=1 for this process so ``endee_client`` matches
    typical LAN installs. Shell or .env may still set ENDEE_LOCAL_NO_AUTH explicitly.
    """
    if not (os.environ.get("ENDEE_BASE_URL") or "").strip():
        return
    if (os.environ.get("ENDEE_API_KEY") or os.environ.get("NDD_AUTH_TOKEN") or "").strip():
        return
    if "ENDEE_LOCAL_NO_AUTH" in os.environ:
        return
    os.environ["ENDEE_LOCAL_NO_AUTH"] = "1"


def _rel_to_repo(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(p.resolve())


def _percentiles_ms(lat: list[float]) -> dict[str, float]:
    if not lat:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    a = np.array(lat, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
    }


def run_qdrant(
    *,
    Q: np.ndarray,
    query_ids: list[str],
    k: int,
    collection: str,
    url: str,
    api_key: str | None,
    hnsw_ef: int,
    warmup: int,
    queries_npz: Path,
) -> tuple[dict, dict]:
    from qdrant_client import QdrantClient
    from qdrant_client.models import SearchParams

    strict = os.environ.get("QDRANT_STRICT_VERSION_CHECK", "").lower() in ("1", "true", "yes")
    client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False, check_compatibility=strict)
    n = Q.shape[0]

    def search_one(vec: np.ndarray) -> tuple[list[dict], float]:
        tq = time.perf_counter()
        resp = client.query_points(
            collection_name=collection,
            query=vec.astype(float).tolist(),
            limit=k,
            search_params=SearchParams(hnsw_ef=hnsw_ef),
            with_payload=True,
        )
        hits = resp.points
        ms = (time.perf_counter() - tq) * 1000.0
        out: list[dict] = []
        for rank, h in enumerate(hits, start=1):
            pl = h.payload or {}
            if not isinstance(pl, dict):
                pl = dict(pl) if pl is not None else {}
            cid = pl.get("_id")
            out.append(
                {
                    "rank": rank,
                    "corpus_id": str(cid) if cid is not None else None,
                    "score": float(h.score) if h.score is not None else None,
                    "title": pl.get("title"),
                }
            )
        return out, ms

    for i in range(min(warmup, n)):
        search_one(Q[i])

    per_query: list[dict] = []
    lat: list[float] = []
    t_wall0 = time.perf_counter()
    for qi in range(n):
        hits, ms = search_one(Q[qi])
        lat.append(ms)
        per_query.append(
            {
                "query_i": qi,
                "query_id": query_ids[qi],
                "latency_ms": round(ms, 4),
                "hits": hits,
            }
        )
    total_wall = time.perf_counter() - t_wall0
    mean_latency_ms = float(np.mean(lat)) if lat else 0.0

    metrics_block = {
        "collection": collection,
        "qdrant_url": url,
        "k": k,
        "num_queries": n,
        "warmup_queries": warmup,
        "hnsw_ef": hnsw_ef,
        "total_wall_seconds": round(total_wall, 4),
        "qps": round(n / total_wall, 4) if total_wall > 0 else 0.0,
        "mean_latency_ms": round(mean_latency_ms, 4),
        "latency_ms": {k2: round(v2, 4) for k2, v2 in _percentiles_ms(lat).items()},
    }

    full_doc = {
        "version": 1,
        "backend": "qdrant",
        "summary": {
            "k": k,
            "num_queries": n,
            "warmup_queries": warmup,
            "hnsw_ef": hnsw_ef,
            "collection": collection,
            "qdrant_url": url,
            "query_vectors_npz": _rel_to_repo(queries_npz),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": per_query,
    }
    return full_doc, metrics_block


def run_endee(
    *,
    Q: np.ndarray,
    query_ids: list[str],
    k: int,
    index_name: str,
    ef: int,
    warmup: int,
    queries_npz: Path,
) -> tuple[dict, dict]:
    client, api_base, auth_mode = resolve_endee_client()
    index = client.get_index(name=index_name)
    n = Q.shape[0]

    def search_one(vec: np.ndarray) -> tuple[list[dict], float]:
        tq = time.perf_counter()
        rows = index.query(
            vector=vec.astype(float).tolist(),
            top_k=k,
            ef=ef,
            include_vectors=False,
        )
        ms = (time.perf_counter() - tq) * 1000.0
        out: list[dict] = []
        for rank, item in enumerate(rows, start=1):
            meta = item.get("meta") or {}
            cid = meta.get("_id")
            sim = item.get("similarity")
            out.append(
                {
                    "rank": rank,
                    "corpus_id": str(cid) if cid is not None else None,
                    "score": float(sim) if sim is not None else None,
                    "title": meta.get("title"),
                }
            )
        return out, ms

    for i in range(min(warmup, n)):
        search_one(Q[i])

    per_query: list[dict] = []
    lat: list[float] = []
    t_wall0 = time.perf_counter()
    for qi in range(n):
        hits, ms = search_one(Q[qi])
        lat.append(ms)
        per_query.append(
            {
                "query_i": qi,
                "query_id": query_ids[qi],
                "latency_ms": round(ms, 4),
                "hits": hits,
            }
        )
    total_wall = time.perf_counter() - t_wall0
    mean_latency_ms = float(np.mean(lat)) if lat else 0.0

    metrics_block = {
        "index_name": index_name,
        "endee_api_base": api_base,
        "endee_auth_mode": auth_mode,
        "k": k,
        "num_queries": n,
        "warmup_queries": warmup,
        "ef": ef,
        "total_wall_seconds": round(total_wall, 4),
        "qps": round(n / total_wall, 4) if total_wall > 0 else 0.0,
        "mean_latency_ms": round(mean_latency_ms, 4),
        "latency_ms": {k2: round(v2, 4) for k2, v2 in _percentiles_ms(lat).items()},
    }

    full_doc = {
        "version": 1,
        "backend": "endee",
        "summary": {
            "k": k,
            "num_queries": n,
            "warmup_queries": warmup,
            "ef": ef,
            "index_name": index_name,
            "endee_api_base": api_base,
            "endee_auth_mode": auth_mode,
            "query_vectors_npz": _rel_to_repo(queries_npz),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": per_query,
    }
    return full_doc, metrics_block


def _write_metrics_retrieval(path: Path, k: int, qdrant: dict | None, endee: dict | None) -> None:
    payload = {
        "version": 1,
        "k": k,
        "note": "Per query: top-k approximate neighbors. k matches ground_truth_top10.json when present.",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if qdrant is not None:
        payload["qdrant"] = qdrant
    if endee is not None:
        payload["endee"] = endee
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    _load_repo_dotenv()
    _apply_endee_local_defaults()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-npz", type=Path, default=BENCHMARKS_DIR / "query_slice.npz")
    parser.add_argument("--k", type=int, default=int(os.environ.get("BENCHMARK_TOP_K", "10")))
    parser.add_argument("--ground-truth", type=Path, default=BENCHMARKS_DIR / "ground_truth_top10.json")
    parser.add_argument("--backend", choices=("qdrant", "endee", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("BENCHMARK_WARMUP", "20")))
    parser.add_argument("--qdrant-ef", type=int, default=int(os.environ.get("QDRANT_HNSW_EF", "128")))
    parser.add_argument("--endee-ef", type=int, default=int(os.environ.get("ENDEE_EF", "128")))
    parser.add_argument("--out-qdrant", type=Path, default=BENCHMARKS_DIR / "qdrant_similarity.json")
    parser.add_argument("--out-endee", type=Path, default=BENCHMARKS_DIR / "endee_similarity.json")
    parser.add_argument("--out-metrics", type=Path, default=BENCHMARKS_DIR / "metrics_retrieval.json")
    args = parser.parse_args()

    k = args.k
    gt_path = args.ground_truth.resolve()
    if gt_path.is_file():
        k = int(json.loads(gt_path.read_text(encoding="utf-8")).get("k", k))

    npz_path = args.queries_npz.resolve()
    if not npz_path.is_file():
        raise SystemExit(f"Missing query npz: {npz_path} — run materialize_query_slice.py first.")

    npz = np.load(npz_path, allow_pickle=True)
    Q = npz["vectors"].astype(np.float32)
    query_ids = [str(x) for x in np.asarray(npz["query_ids"]).tolist()]

    metrics_q: dict | None = None
    metrics_e: dict | None = None

    if args.backend in ("qdrant", "both"):
        url = os.environ.get("QDRANT_URL", "http://192.168.108.123:6333").rstrip("/")
        coll = os.environ.get("QDRANT_COLLECTION", "dbpedia_10k_benchmark_native")
        api_key = os.environ.get("QDRANT_API_KEY") or None
        doc, metrics_q = run_qdrant(
            Q=Q,
            query_ids=query_ids,
            k=k,
            collection=coll,
            url=url,
            api_key=api_key,
            hnsw_ef=args.qdrant_ef,
            warmup=args.warmup,
            queries_npz=npz_path,
        )
        out_q = args.out_qdrant.resolve()
        out_q.parent.mkdir(parents=True, exist_ok=True)
        out_q.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_q}")

    if args.backend in ("endee", "both"):
        idx = os.environ.get("ENDEE_INDEX_NAME", "dbpedia_10k_benchmark")
        doc, metrics_e = run_endee(
            Q=Q,
            query_ids=query_ids,
            k=k,
            index_name=idx,
            ef=args.endee_ef,
            warmup=args.warmup,
            queries_npz=npz_path,
        )
        out_e = args.out_endee.resolve()
        out_e.parent.mkdir(parents=True, exist_ok=True)
        out_e.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_e}")

    out_m = args.out_metrics.resolve()
    _write_metrics_retrieval(out_m, k, metrics_q, metrics_e)
    print(f"Wrote {out_m}")


if __name__ == "__main__":
    main()
