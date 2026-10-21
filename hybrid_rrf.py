import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from evaluate_retrieval import load_labels, load_rankings


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "lotte" / "lifestyle" / "dev"
BASE = ROOT / "artifacts" / "lifestyle" / "dev"
OUTPUT = BASE / "hybrid"

METHODS = ("tfidf", "bm25", "semantic")
RRF_K = 60
CANDIDATE_LIMIT = 100


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels(DATA)

    # Labels supply the query IDs and validate the ranking files.
    # Relevant passage labels are never used in fusion.
    runs = {}
    for method in METHODS:
        path = BASE / method / "rankings.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Missing rankings: {path}")

        runs[method] = load_rankings(path, labels)
        print(f"Loaded {method} rankings")

    fusion_times = []
    candidate_counts = []
    diagnostics = []

    with (OUTPUT / "rankings.tsv").open(
        "w", encoding="utf-8"
    ) as output:
        for qid in tqdm(sorted(labels), desc="Combining rankings"):
            start = time.perf_counter()
            combined_scores = defaultdict(float)

            for method in METHODS:
                for rank, pid in enumerate(runs[method][qid], start=1):
                    combined_scores[pid] += 1.0 / (RRF_K + rank)

            ranked = sorted(
                combined_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            top = ranked[:CANDIDATE_LIMIT]

            elapsed = time.perf_counter() - start
            fusion_times.append(elapsed)
            candidate_counts.append(len(combined_scores))

            for rank, (pid, score) in enumerate(top, start=1):
                output.write(f"{qid}\t{pid}\t{rank}\t{score:.12g}\n")

            diagnostics.append({
                "query_id": qid,
                "unique_candidates": len(combined_scores),
                "returned": len(top),
                "fusion_latency_ms": elapsed * 1000,
            })

    metrics = {
        "method": "hybrid",
        "domain": "lifestyle",
        "split": "dev",
        "query_type": "search",
        "queries": len(labels),
        "component_methods": list(METHODS),
        "fusion": "reciprocal_rank_fusion",
        "rrf_k": RRF_K,
        "weights": {method: 1.0 for method in METHODS},
        "candidates_per_method": 100,
        "candidate_limit": CANDIDATE_LIMIT,
        "mean_unique_candidates": float(np.mean(candidate_counts)),
        "mean_fusion_latency_ms": float(np.mean(fusion_times) * 1000),
        "mean_query_latency_ms": None,
        "latency_note": (
            "Offline fusion time only. End-to-end hybrid query latency "
            "must be measured when all retrievers run together."
        ),
    }

    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (OUTPUT / "fusion_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    print("\nHybrid RRF complete")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved files: {OUTPUT}")


if __name__ == "__main__":
    main()