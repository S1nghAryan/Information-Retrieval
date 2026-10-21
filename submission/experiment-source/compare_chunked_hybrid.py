"""Evaluate two previously selected RRF weights with chunked semantic runs."""
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from evaluate_retrieval import load_labels, load_rankings, evaluate
from tune_rrf import fuse, save_result

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts' / 'lifestyle' / 'dev'
DATA = ROOT / 'data' / 'lotte' / 'lifestyle' / 'dev'
KEYS = ('precision_at_10', 'success_at_5', 'recall_at_10',
        'recall_at_100', 'map_at_100', 'ndcg_at_10')
CONFIGS = (
    ('hybrid_chunked_three', (0.25, 1.0, 4.0)),
    ('hybrid_chunked_two', (0.0, 1.0, 4.0)),
)


def main():
    labels = load_labels(DATA)
    runs, hashes = {}, {}
    # fuse() uses the logical name 'semantic'; explicitly map it to chunked.
    for key, folder in (('tfidf', 'tfidf'), ('bm25', 'bm25'),
                        ('semantic', 'semantic_chunked')):
        path = BASE / folder / 'rankings.tsv'
        if not path.exists():
            raise FileNotFoundError(f'Missing input rankings: {path}')
        runs[key] = load_rankings(path, labels)
        hashes[folder] = hashlib.sha256(path.read_bytes()).hexdigest()

    # Recompute references using the same evaluator, rather than copied values.
    rows = []
    for name in ('semantic', 'semantic_chunked', 'hybrid_tuned', 'hybrid_two_way'):
        path = BASE / name / 'rankings.tsv'
        ranking = load_rankings(path, labels)
        _, summary, _ = evaluate(labels, ranking)
        rows.append({'method': name, **{key: summary[key] for key in KEYS}})

    details = []
    for name, weights in CONFIGS:
        start = time.perf_counter()
        scored, ranking = fuse(runs, labels, weights)
        elapsed_ms = (time.perf_counter() - start) * 1000 / len(labels)
        per_query, summary, curve = evaluate(labels, ranking)
        metadata = {
            'method': name, 'domain': 'lifestyle', 'split': 'dev',
            'query_type': 'search', 'rrf_k': 60,
            'weights': dict(zip(('tfidf', 'bm25', 'semantic_chunked'), weights)),
            'semantic_variant': 'semantic_chunked',
            'candidate_limit': 100, 'candidates_per_method': 100,
            'selection_status': 'fixed earlier weights; no additional weight search',
            'mean_fusion_latency_ms': elapsed_ms,
            'mean_query_latency_ms': None,
            'latency_note': 'Offline fusion only, not full search latency.',
            'source_ranking_sha256': hashes,
        }
        save_result(BASE / name, scored, per_query, summary, curve, metadata)
        rows.append({'method': name, **{key: summary[key] for key in KEYS}})
        details.append({**metadata, **summary})
        print(f"{name}: nDCG@10={summary['ndcg_at_10']:.6f}", flush=True)

    table = pd.DataFrame(rows)
    output = BASE / 'chunked_hybrid_comparison'
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / 'comparison.csv', index=False)
    by_name = table.set_index('method')
    changes = []
    for new, old in (('semantic_chunked', 'semantic'),
                     ('hybrid_chunked_three', 'hybrid_tuned'),
                     ('hybrid_chunked_two', 'hybrid_two_way')):
        changes.append({'comparison': f'{new} minus {old}', **{
            key: float(by_name.loc[new, key] - by_name.loc[old, key])
            for key in KEYS}})
    pd.DataFrame(changes).to_csv(output / 'deltas.csv', index=False)
    print('\nDEV COMPARISON')
    print(table.to_string(index=False, float_format=lambda value: f'{value:.6f}'))
    print('\nCHANGE CAUSED BY REPLACING SEMANTIC RANKINGS (same weights)')
    print(pd.DataFrame(changes).to_string(
        index=False, float_format=lambda value: f'{value:+.6f}'))
    print(f'\nSaved comparison: {output}')
    print('Original rankings preserved. Test set not accessed.')
    stamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d %H:%M IST')
    with (ROOT / 'processss.txt').open('a', encoding='utf-8') as log:
        log.write(f'\n\nFIXED-WEIGHT CHUNKED HYBRID COMPARISON | {stamp}\n')
        log.write('What: Replaced semantic with chunked rankings at fixed k=60 and weights.\n')
        log.write('Why: Isolate the effect of chunking without extending the tuning grid.\n')
        log.write(json.dumps({'results': details, 'comparison': rows, 'deltas': changes}, indent=2))
        log.write('\nStatus: Dev comparison completed; final configuration not automatically frozen.\n')
    print('Appended results to local processss.txt.')


if __name__ == '__main__':
    main()
