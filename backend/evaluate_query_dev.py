"""Explore query preprocessing in existing methods, using development data only.

No indexes, deployment settings or dashboard scores are changed by this script.
The previously used dev split is exploratory, not an independent holdout.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import Stemmer
import math

KEYS = ['ndcg10', 'precision10', 'recall10', 'recall100', 'map100', 'success3']


def metrics(pids, relevant):
    hits = [int(pid in relevant) for pid in pids]
    ap = sum(sum(hits[:i+1])/(i+1) for i, hit in enumerate(hits) if hit)/len(relevant)
    dcg = sum(hit/math.log2(i+2) for i, hit in enumerate(hits[:10]))
    ideal = sum(1/math.log2(i+2) for i in range(min(10, len(relevant))))
    return dict(zip(KEYS, [dcg/ideal, sum(hits[:10])/10, sum(hits[:10])/len(relevant),
                           sum(hits)/len(relevant), ap, float(any(hits[:3]))]))



ROOT = Path(os.environ.get('RAG_ROOT', str(Path(__file__).resolve().parents[1])))
BASE = ROOT / 'artifacts/lifestyle/dev'
DATA = ROOT / 'data/lotte/lifestyle/dev'
OUT = Path(__file__).resolve().parent.parent / 'submission'
NOISE = set('what which how why when where who is are can could would should do does did i we you my our your please'.split())


def main():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stemmer = Stemmer.Stemmer('english')
    queries = {int(q): text for q, text in
               (line.split('\t', 1) for line in (DATA/'questions.search.tsv').read_text(encoding='utf-8').splitlines())}
    policies = {'baseline': set(), 'question_words': NOISE, 'stopwords': set(ENGLISH_STOP_WORDS) - {'not', 'no', 'never'}}
    protocol = {
        'scope': 'Exploratory development only; dev was used in earlier tuning. No test labels or rankings are read.',
        'policies': {key: sorted(value) for key, value in policies.items()},
        'selection_rule': 'Highest selection nDCG@10; keep baseline on ties. Report validation once for each selected policy.',
        'promotion_rule': 'Do not deploy automatically; an independent validation set is needed for a new generalization claim.',
        'query_sha256': hashlib.sha256((DATA/'questions.search.tsv').read_bytes()).hexdigest(),
    }
    (OUT/'query-experiment-protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    processed = {}
    for policy, stops in policies.items():
        processed[policy] = {}
        for qid, text in queries.items():
            words = re.findall(r'\b\w+\b', text.lower())
            kept = [word for word in words if word not in stops]
            processed[policy][qid] = ' '.join(stemmer.stemWords(kept or words))
    predictions = {}
    for method in ['tfidf', 'bm25']:
        pids = np.load(BASE/method/'passage_ids.npy', allow_pickle=False)
        if method == 'tfidf':
            import joblib
            from scipy.sparse import load_npz
            vectorizer = joblib.load(BASE/method/'vectorizer.joblib')
            postings = load_npz(BASE/method/'matrix.npz').T.tocsr()
        else:
            import bm25s
            retriever = bm25s.BM25.load(str(BASE/method/'index'), mmap=True)
        predictions[method] = {}
        for policy in policies:
            runs = {}
            for qid, text in processed[policy].items():
                if method == 'tfidf':
                    scores = (vectorizer.transform([text]) @ postings).toarray().ravel()
                else:
                    tokens = [t for t in text.split() if t in retriever.vocab_dict]
                    scores = retriever.get_scores(tokens) if tokens else np.zeros(len(pids))
                matches = np.flatnonzero(scores > 0)
                order = np.lexsort((pids[matches], -scores[matches]))
                runs[qid] = pids[matches[order[:100]]].tolist()
            predictions[method][policy] = runs
            print(f'{method}: {policy} retrieved {len(runs)} queries', flush=True)
        if method == 'tfidf':
            del vectorizer, postings
        else:
            del retriever
        gc.collect()
    semantic = {qid: [] for qid in queries}
    for line in (BASE/'semantic_chunked/rankings.tsv').read_text().splitlines():
        qid, pid, rank, _ = line.split()
        semantic[int(qid)].append((int(rank), int(pid)))
    semantic = {qid: [pid for _, pid in sorted(rows)] for qid, rows in semantic.items()}
    predictions['hybrid_chunked_two'] = {}
    for policy in policies:
        runs = {}
        for qid in queries:
            fused = {}
            for ranking, weight in [(predictions['bm25'][policy][qid], 1), (semantic[qid], 4)]:
                for rank, pid in enumerate(ranking, 1):
                    fused[pid] = fused.get(pid, 0) + weight/(60+rank)
            runs[qid] = sorted(fused, key=lambda pid: (-fused[pid], pid))[:100]
        predictions['hybrid_chunked_two'][policy] = runs
    # Labels are consulted only after all candidate rankings are produced.
    labels = {int(row['qid']): set(row['answer_pids']) for row in
              map(json.loads, (DATA/'qas.search.jsonl').read_text().splitlines())}
    ids = sorted(labels, key=lambda q: hashlib.sha256(f'fieldnotes-fusion-dev-v1:{q}'.encode()).digest())
    selection, validation = ids[:333], ids[333:]
    report = {**protocol, 'selection_qids': selection, 'validation_qids': validation, 'methods': {}}
    for method, variants in predictions.items():
        per_query = {policy: {qid: metrics(runs[qid], labels[qid]) for qid in ids}
                     for policy, runs in variants.items()}
        def average(policy, subset):
            return {key: sum(per_query[policy][qid][key] for qid in subset)/len(subset) for key in KEYS}
        saved = json.loads((BASE/method/'metrics.json').read_text())
        assert abs(average('baseline', ids)['ndcg10'] - saved['ndcg_at_10']) < 1e-8
        selection_scores = {policy: average(policy, selection) for policy in policies}
        winner = max(policies, key=lambda p: (selection_scores[p]['ndcg10'], p == 'baseline'))
        report['methods'][method] = {
            'selection': selection_scores, 'selected_policy': winner,
            'baseline_validation': average('baseline', validation),
            'candidate_validation': average(winner, validation),
            'baseline_full_dev': average('baseline', ids),
            'candidate_full_dev': average(winner, ids),
        }
    report['live_settings_changed'] = False
    (OUT/'query-experiment-dev.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    text = '# Existing-method improvement experiment\n\n' + protocol['scope'] + '\n\n'
    text += 'Three fixed query policies: unchanged, remove common question words, remove English stopwords while retaining not/no/never. Empty filtered queries fall back to original terms. Passage indexes, stemming and ranking formulas stay fixed. The semantic ranking stays unchanged; RRF uses the candidate BM25 ranking with semantic weight 4 and k=60.\n\n'
    text += '| Method | Selected query policy | Baseline validation nDCG@10 | Candidate validation nDCG@10 | Baseline R@10 | Candidate R@10 |\n|---|---|---:|---:|---:|---:|\n'
    for method, row in report['methods'].items():
        b, c = row['baseline_validation'], row['candidate_validation']
        text += f"| {method} | {row['selected_policy']} | {b['ndcg10']:.6f} | {c['ndcg10']:.6f} | {b['recall10']:.6f} | {c['recall10']:.6f} |\n"
    text += '\nNo automatic promotion or dashboard replacement. This split was previously inspected and is not fresh held-out evidence. Baseline scores reproduce the original evaluator. Exact policy lists, split IDs and selection/validation results are in query-experiment-dev.json.\n'
    (OUT/'query-experiment-dev.md').write_text(text, encoding='utf-8')
    print(text, flush=True)


if __name__ == '__main__':
    main()
