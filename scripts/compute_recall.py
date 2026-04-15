#!/usr/bin/env python3
"""
Phase 4 — Recall@k from ground truth vs ANN JSON exports (no live DB).

Reads ground_truth_top10.json plus optional qdrant_similarity.json / endee_similarity.json.
Writes benchmarks/recall_results.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


def _truth_ids(gt_result: dict, k: int) -> set[str]:
    return {n["corpus_id"] for n in gt_result["top10"][:k]}


def _pred_ids(ann_result: dict, k: int) -> set[str]:
    out: set[str] = set()
    for h in ann_result.get("hits", [])[:k]:
        cid = h.get("corpus_id")
        if cid is not None:
            out.add(str(cid))
    return out


def _recall_one(truth: set[str], pred: set[str]) -> float:
    if not truth:
        return 0.0
    return len(pred & truth) / len(truth)


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=BENCHMARKS_DIR / "ground_truth_top10.json")
    parser.add_argument("--qdrant", type=Path, default=BENCHMARKS_DIR / "qdrant_similarity.json")
    parser.add_argument("--endee", type=Path, default=BENCHMARKS_DIR / "endee_similarity.json")
    parser.add_argument("--output", type=Path, default=BENCHMARKS_DIR / "recall_results.json")
    args = parser.parse_args()

    gt_path = args.ground_truth.resolve()
    if not gt_path.is_file():
        raise SystemExit(f"Missing ground truth: {gt_path}")

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    k = int(gt.get("k", 10))
    n = int(gt.get("num_queries", len(gt.get("results", []))))
    gt_results = gt["results"]
    if len(gt_results) != n:
        n = len(gt_results)

    qdrant_doc = _load_json(args.qdrant.resolve())
    endee_doc = _load_json(args.endee.resolve())

    out: dict = {
        "version": 1,
        "k": k,
        "num_queries": n,
        "ground_truth_json": str(gt_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backends": {},
    }

    def eval_backend(name: str, doc: dict | None, source_path: Path) -> None:
        if doc is None:
            out["backends"][name] = {"skipped": True, "reason": "file not found"}
            return
        rows = doc.get("results", [])
        if len(rows) != n:
            out["backends"][name] = {
                "error": f"result count {len(rows)} != ground_truth num_queries {n}",
            }
            return
        recalls: list[float] = []
        per_query: list[dict] = []
        for qi in range(n):
            gid = gt_results[qi].get("query_id")
            rid = rows[qi].get("query_id")
            if gid is not None and rid is not None and str(gid) != str(rid):
                raise SystemExit(
                    f"{name}: query_id mismatch at query_i={qi}: ground_truth={gid!r} ann={rid!r}",
                )
            truth = _truth_ids(gt_results[qi], k)
            pred = _pred_ids(rows[qi], k)
            r = _recall_one(truth, pred)
            recalls.append(r)
            per_query.append(
                {"query_i": qi, "query_id": gt_results[qi].get("query_id"), "recall_at_k": round(r, 6)},
            )
        out["backends"][name] = {
            "source_json": str(source_path),
            "recall_at_k_mean": round(sum(recalls) / len(recalls), 6) if recalls else 0.0,
            "recall_at_k_per_query": [round(x, 6) for x in recalls],
            "per_query": per_query,
        }

    eval_backend("qdrant", qdrant_doc, args.qdrant.resolve())
    eval_backend("endee", endee_doc, args.endee.resolve())

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
