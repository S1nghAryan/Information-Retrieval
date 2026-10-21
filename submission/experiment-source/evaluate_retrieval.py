import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_labels(folder):
    query_ids = set()

    with (folder / "questions.search.tsv").open(encoding="utf-8") as file:
        for line in file:
            qid, _ = line.rstrip("\r\n").split("\t", 1)
            query_ids.add(int(qid))

    labels = {}
    with (folder / "qas.search.jsonl").open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            qid = int(item["qid"])
            if qid in labels:
                raise ValueError(f"Duplicate labels for query {qid}")
            labels[qid] = {int(pid) for pid in item["answer_pids"]}

    if set(labels) != query_ids:
        raise ValueError("Query IDs and label IDs do not match.")
    if any(not relevant for relevant in labels.values()):
        raise ValueError("A query has no relevant passage labels.")

    return labels


def load_rankings(path, labels):
    # Include every query, even if retrieval returned no results.
    grouped = {qid: [] for qid in labels}

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"Invalid ranking row at line {line_number}")

            qid, pid, rank = map(int, fields[:3])
            score = float(fields[3])

            if qid not in grouped:
                raise ValueError(f"Unknown query ID: {qid}")
            if rank < 1 or not np.isfinite(score):
                raise ValueError(f"Invalid rank or score at line {line_number}")

            grouped[qid].append((rank, pid))

    results = {}
    for qid, rows in grouped.items():
        rows.sort()
        ranks = [rank for rank, _ in rows]
        pids = [pid for _, pid in rows]

        if ranks != list(range(1, len(rows) + 1)):
            raise ValueError(f"Nonsequential or duplicate ranks for query {qid}")
        if len(pids) != len(set(pids)):
            raise ValueError(f"Duplicate passage results for query {qid}")

        results[qid] = pids[:100]

    return results


def evaluate(labels, rankings):
    records = []
    recall_grid = np.linspace(0, 1, 101)
    curves = []

    for qid in sorted(labels):
        relevant = labels[qid]
        retrieved = rankings[qid]
        total_relevant = len(relevant)

        # Missing result positions count as nonrelevant.
        hits = np.zeros(100, dtype=np.float64)
        for index, pid in enumerate(retrieved):
            hits[index] = float(pid in relevant)

        cumulative_hits = np.cumsum(hits)
        positions = np.arange(1, 101)
        precision = cumulative_hits / positions
        recall = cumulative_hits / total_relevant

        # AP@100 uses ALL known relevant passages as its denominator.
        ap100 = float(np.sum(precision * hits) / total_relevant)

        discounts = 1 / np.log2(np.arange(2, 12))
        dcg10 = float(np.sum(hits[:10] * discounts))
        ideal_dcg10 = float(np.sum(discounts[:min(total_relevant, 10)]))

        records.append({
            "query_id": qid,
            "relevant_passages": total_relevant,
            "returned_at_100": len(retrieved),
            "precision_at_10": float(cumulative_hits[9] / 10),
            "success_at_5": float(cumulative_hits[4] > 0),
            "recall_at_10": float(recall[9]),
            "recall_at_100": float(recall[99]),
            "ap_at_100": ap100,
            "ndcg_at_10": dcg10 / ideal_dcg10,
        })

        # Interpolated PR curve using only the retrieved top 100.
        # Unreached recall levels receive precision 0.
        curves.append([
            float(precision[recall >= level].max())
            if np.any(recall >= level) else 0.0
            for level in recall_grid
        ])

    table = pd.DataFrame(records)

    summary = {
        "evaluated_queries": len(table),
        "queries_with_no_results": int(
            (table["returned_at_100"] == 0).sum()
        ),
        "precision_at_10": float(table["precision_at_10"].mean()),
        "success_at_5": float(table["success_at_5"].mean()),
        "recall_at_10": float(table["recall_at_10"].mean()),
        "recall_at_100": float(table["recall_at_100"].mean()),
        "map_at_100": float(table["ap_at_100"].mean()),
        "ndcg_at_10": float(table["ndcg_at_10"].mean()),
        "ap_denominator": "all labeled relevant passages",
        "relevance": "binary",
        "pr_curve": "101-point interpolated, retrieval truncated at 100",
    }

    curve = pd.DataFrame({
        "recall": recall_grid,
        "interpolated_precision": np.mean(curves, axis=0),
    })

    return table, summary, curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="tfidf")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    data = ROOT / "data" / "lotte" / "lifestyle" / args.split
    output = ROOT / "artifacts" / "lifestyle" / args.split / args.method
    ranking_path = output / "rankings.tsv"

    if not ranking_path.exists():
        raise FileNotFoundError(f"Ranking file not found: {ranking_path}")

    labels = load_labels(data)
    rankings = load_rankings(ranking_path, labels)
    per_query, summary, curve = evaluate(labels, rankings)

    # Preserve timing results recorded by the retrieval script.
    metrics_path = output / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists() else {}
    )
    metrics.update(summary)
    metrics.update({
        "method": args.method,
        "domain": "lifestyle",
        "split": args.split,
        "query_type": "search",
    })

    metrics_path.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
    per_query.to_csv(output / "per_query_metrics.csv", index=False)
    curve.to_csv(output / "precision_recall.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        curve["recall"],
        curve["interpolated_precision"],
        label=args.method.upper(),
        linewidth=2,
    )
    ax.set(
        xlabel="Recall",
        ylabel="Interpolated precision",
        title=f"LoTTE Lifestyle {args.split}: top-100 retrieval",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_recall.png", dpi=180)
    plt.close(fig)

    print(f"\n{args.method.upper()} — {args.split.upper()}")
    print(f"Evaluated queries: {len(labels)}")
    for key in (
        "precision_at_10",
        "success_at_5",
        "recall_at_10",
        "recall_at_100",
        "map_at_100",
        "ndcg_at_10",
    ):
        print(f"{key:22s}: {summary[key]:.6f}")

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()