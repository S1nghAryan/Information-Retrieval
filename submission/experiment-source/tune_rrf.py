"""Compare a fixed, small RRF grid on LoTTE Lifestyle dev only."""
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from evaluate_retrieval import load_labels, load_rankings, evaluate

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts' / 'lifestyle' / 'dev'
DATA = ROOT / 'data' / 'lotte' / 'lifestyle' / 'dev'
METHODS = ('tfidf', 'bm25', 'semantic')
RRF_K = 60
# Fixed before observing results. Tuple order: TF-IDF, BM25, semantic.
CONFIGS = [
    ('equal_three', (1.0, 1.0, 1.0)),
    ('two_equal', (0.0, 1.0, 1.0)),
    ('two_semantic_2', (0.0, 1.0, 2.0)),
    ('two_semantic_4', (0.0, 1.0, 4.0)),
    ('three_light_2', (0.25, 1.0, 2.0)),
    ('three_light_4', (0.25, 1.0, 4.0)),
    ('three_medium_2', (0.5, 1.0, 2.0)),
    ('three_medium_4', (0.5, 1.0, 4.0)),
]


def fuse(runs, qids, weights):
    scored, ranking = {}, {}
    for qid in sorted(qids):
        scores = defaultdict(float)
        for method, weight in zip(METHODS, weights):
            if weight == 0:
                continue
            for rank, pid in enumerate(runs[method][qid], 1):
                scores[pid] += weight / (RRF_K + rank)
        scored[qid] = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:100]
        ranking[qid] = [pid for pid, _ in scored[qid]]
    return scored, ranking


def save_result(folder, scored, per_query, summary, curve, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'rankings.tsv').open('w', encoding='utf-8') as file:
        for qid in sorted(scored):
            for rank, (pid, score) in enumerate(scored[qid], 1):
                file.write(f'{qid}\t{pid}\t{rank}\t{score:.12g}\n')
    metrics = {**summary, **metadata}
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    pd.DataFrame([metrics]).to_csv(folder / 'metrics.csv', index=False)
    per_query.to_csv(folder / 'per_query_metrics.csv', index=False)
    curve.to_csv(folder / 'precision_recall.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve['recall'], curve['interpolated_precision'])
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Recall',
           ylabel='Interpolated precision',
           title=f"{metadata['method']}: Lifestyle dev, top 100")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(folder / 'precision_recall.png', dpi=180)
    plt.close(fig)


def main():
    tuning = BASE / 'rrf_tuning'
    tuning.mkdir(parents=True, exist_ok=True)
    labels = load_labels(DATA)
    runs, fingerprints = {}, {}
    for method in METHODS:
        path = BASE / method / 'rankings.tsv'
        runs[method] = load_rankings(path, labels)
        fingerprints[method] = hashlib.sha256(path.read_bytes()).hexdigest()

    _, semantic_summary, _ = evaluate(labels, runs['semantic'])
    rows, results = [], {}
    for name, weights in CONFIGS:
        start = time.perf_counter()
        scored, ranking = fuse(runs, labels, weights)
        fusion_ms = (time.perf_counter() - start) * 1000 / len(labels)
        per_query, summary, curve = evaluate(labels, ranking)
        metadata = {
            'method': name, 'domain': 'lifestyle', 'split': 'dev',
            'query_type': 'search', 'rrf_k': RRF_K,
            'weights': dict(zip(METHODS, weights)), 'candidate_limit': 100,
            'candidates_per_method': 100,
            'mean_fusion_latency_ms': fusion_ms,
            'mean_query_latency_ms': None,
            'latency_note': 'Offline fusion only; not end-to-end retrieval.',
            'selection_metric': 'ndcg_at_10',
            'source_ranking_sha256': fingerprints,
        }
        result = (scored, per_query, summary, curve, metadata)
        results[name] = result
        save_result(tuning / name, *result)
        rows.append({
            'configuration': name, 'tfidf_weight': weights[0],
            'bm25_weight': weights[1], 'semantic_weight': weights[2],
            **{key: summary[key] for key in (
                'precision_at_10', 'success_at_5', 'recall_at_10',
                'recall_at_100', 'map_at_100', 'ndcg_at_10')},
        })
        print(f"{name}: nDCG@10={summary['ndcg_at_10']:.6f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(tuning / 'comparison.csv', index=False)
    # max() retains predefined configuration order for exact metric ties.
    best_three = max((r for r in rows if r['tfidf_weight'] > 0),
                     key=lambda r: r['ndcg_at_10'])
    best_two = max((r for r in rows if r['tfidf_weight'] == 0),
                   key=lambda r: r['ndcg_at_10'])
    selection = {
        'status': 'provisional dev selection; not final test results',
        'selection_metric': 'ndcg_at_10',
        'tie_break': 'first in predefined configuration order',
        'rrf_k': RRF_K, 'semantic_ndcg_at_10': semantic_summary['ndcg_at_10'],
        'best_three_method': best_three, 'best_two_method': best_two,
        'source_ranking_sha256': fingerprints,
    }
    for selected, name in ((best_three, 'hybrid_tuned'), (best_two, 'hybrid_two_way')):
        scored, per_query, summary, curve, metadata = results[selected['configuration']]
        metadata = {**metadata, 'method': name,
                    'selected_configuration': selected['configuration'],
                    'selection_status': 'selected on dev; test not evaluated'}
        save_result(BASE / name, scored, per_query, summary, curve, metadata)
    (tuning / 'selected_config.json').write_text(
        json.dumps(selection, indent=2), encoding='utf-8')

    print('\nDEV COMPARISON (descending nDCG@10)')
    print(table.sort_values('ndcg_at_10', ascending=False, kind='stable').to_string(
        index=False, float_format=lambda x: f'{x:.6f}'))
    print(f"\nSemantic alone nDCG@10: {semantic_summary['ndcg_at_10']:.6f}")
    print('Best three-method:', best_three['configuration'])
    print('Best two-method:', best_two['configuration'])
    print('\nOriginal hybrid results preserved. No test data used.')
    print(f'Results: {tuning}')

    timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d %H:%M IST')
    with (ROOT / 'processss.txt').open('a', encoding='utf-8') as log:
        log.write(f'\n\nRRF TUNING EXECUTED | {timestamp}\n')
        log.write('What: Compared eight predefined RRF settings on dev; k=60.\n')
        log.write('Why: Equal-weight fusion underperformed semantic retrieval.\n')
        log.write('Selection: highest dev nDCG@10; predefined order breaks ties.\n')
        log.write(json.dumps(selection, indent=2) + '\n')
        log.write('Status: provisional. Preserve original results; do not tune on test.\n')
    print('Updated local processss.txt with measured tuning results.')


if __name__ == '__main__':
    main()
