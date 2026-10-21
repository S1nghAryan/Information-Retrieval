"""Frozen LoTTE Lifestyle test pipeline. No tuning or model training."""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from itertools import islice
from pathlib import Path

import bm25s
import faiss
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import save_npz
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from tfidf_baseline import preprocess
from semantic_chunked import read_tsv, split_long, search_unique, hash_file
from evaluate_retrieval import load_labels, load_rankings
from tune_rrf import fuse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'lotte' / 'lifestyle' / 'test'
OUT = ROOT / 'artifacts' / 'lifestyle' / 'test'
MODEL = ROOT / 'artifacts' / 'lifestyle' / 'dev' / 'semantic' / 'embedding_model'
METHODS = ('tfidf', 'bm25', 'semantic_chunked',
           'hybrid_chunked_three', 'hybrid_chunked_two')
TOTAL = 119461
CONFIG = {
    'domain': 'lifestyle', 'split': 'test', 'query_type': 'search',
    'lexical_preprocessing': 'lowercase, regex word tokens, English stemming, no stopword removal',
    'tfidf': {'sublinear_tf': True, 'smooth_idf': True, 'norm': 'l2', 'dtype': 'float32'},
    'bm25': {'method': 'lucene', 'k1': 1.5, 'b': 0.75},
    'semantic': {'model': 'intfloat/e5-small-v2', 'input_limit': 512,
                 'long_passage_window': 480, 'overlap': 64,
                 'aggregation': 'max chunk similarity', 'query_device': 'cpu'},
    'rrf_k': 60, 'candidates_per_method': 100,
    'three_method_weights': [0.25, 1.0, 4.0],
    'two_method_weights': [0.0, 1.0, 4.0],
    'settings_selected_on': 'dev nDCG@10',
}


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(path)


def log_entry(title, details):
    stamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d %H:%M IST')
    with (ROOT / 'processss.txt').open('a', encoding='utf-8') as file:
        file.write(f'\n\n{title} | {stamp}\n')
        file.write(json.dumps(details, indent=2) + '\n')


def initialize():
    for path in (DATA / 'collection.tsv', DATA / 'questions.search.tsv',
                 DATA / 'qas.search.jsonl', MODEL / 'modules.json',
                 ROOT / 'evaluate_retrieval.py'):
        if not path.exists():
            raise FileNotFoundError(f'Required file missing: {path}')
    OUT.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256()
    for path in sorted(p for p in MODEL.rglob('*') if p.is_file()):
        model_hash.update(str(path.relative_to(MODEL)).encode())
        model_hash.update(hash_file(path).encode())
    # Hash helpers, data, and model to prevent mixing incompatible checkpoints.
    manifest = {**CONFIG, 'model_sha256': model_hash.hexdigest(),
        'collection_sha256': hash_file(DATA / 'collection.tsv'),
        'queries_sha256': hash_file(DATA / 'questions.search.tsv'),
        'qrels_sha256': hash_file(DATA / 'qas.search.jsonl'),
        'implementation_sha256': {name: hash_file(ROOT / name) for name in (
            'run_test_evaluation.py', 'tfidf_baseline.py', 'semantic_chunked.py',
            'tune_rrf.py', 'evaluate_retrieval.py')},
        'versions': {name: version(name) for name in (
            'torch', 'sentence-transformers', 'transformers', 'bm25s',
            'scikit-learn', 'faiss-cpu', 'numpy')},
    }
    target = OUT / 'frozen_config.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Frozen configuration changed. Stop and inspect; do not delete checkpoints.')
    else:
        write_json(target, manifest)
        log_entry('FROZEN TEST RUN PREPARED', {
            'what': 'Index separate test corpus using dev-selected settings.',
            'why': 'Held-out evaluation; no test-label-based tuning.', 'config': manifest})
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def finish(folder, name, metrics, run_key):
    write_json(folder / 'metrics.json', {
        'method': name, 'domain': 'lifestyle', 'split': 'test',
        'query_type': 'search', 'candidate_limit': 100, **metrics})
    write_json(folder / 'completed.json', {'run_key': run_key})


def completed(folder, run_key):
    marker = folder / 'completed.json'
    if marker.exists():
        if json.loads(marker.read_text())['run_key'] != run_key:
            raise ValueError(f'Incompatible completed stage: {folder}')
        if not (folder / 'rankings.tsv').exists():
            raise FileNotFoundError(f'Completed stage missing rankings: {folder}')
        print(f'Reusing completed stage: {folder.name}', flush=True)
        return True
    folder.mkdir(parents=True, exist_ok=True)
    return False


def lexical(name, queries, run_key):
    folder = OUT / name
    if completed(folder, run_key):
        return
    print(f'\nBuilding test {name} index...', flush=True)
    ids = []
    start = time.perf_counter()

    def documents():
        for pid, text in tqdm(read_tsv(DATA / 'collection.tsv'), total=TOTAL, desc=name):
            ids.append(pid)
            yield preprocess(text)

    if name == 'tfidf':
        vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None,
            lowercase=False, norm='l2', sublinear_tf=True, smooth_idf=True, dtype=np.float32)
        matrix = vectorizer.fit_transform(documents())
        build_seconds = time.perf_counter() - start
        save_npz(folder / 'matrix.npz', matrix)
        joblib.dump(vectorizer, folder / 'vectorizer.joblib')
        postings = matrix.T.tocsr()
        del matrix
        def score_query(text):
            return (vectorizer.transform([preprocess(text)]) @ postings).toarray().ravel()
    else:
        texts = list(documents())
        tokens = bm25s.tokenize(texts, stopwords=[], stemmer=None,
                               token_pattern=r'(?u)\b\w+\b')
        del texts
        retriever = bm25s.BM25(k1=1.5, b=0.75, method='lucene')
        retriever.index(tokens)
        build_seconds = time.perf_counter() - start
        del tokens
        retriever.save(str(folder / 'index'))
        def score_query(text):
            words = [w for w in preprocess(text).split() if w in retriever.vocab_dict]
            return retriever.get_scores(words) if words else np.zeros(len(ids), dtype=np.float32)

    pids = np.asarray(ids, dtype=np.int64)
    if len(pids) != TOTAL or len(set(ids)) != TOTAL:
        raise ValueError('Unexpected test passage count or duplicate IDs.')
    np.save(folder / 'passage_ids.npy', pids)
    score_query('warmup query')
    times = []
    with (folder / 'rankings.tsv').open('w', encoding='utf-8') as file:
        for qid, text in tqdm(queries, desc=f'Test {name} search'):
            start = time.perf_counter()
            scores = score_query(text)
            candidates = np.flatnonzero(scores > 0)
            top = candidates[np.lexsort((pids[candidates], -scores[candidates]))[:100]]
            times.append(time.perf_counter() - start)
            for rank, pos in enumerate(top, 1):
                file.write(f'{qid}\t{int(pids[pos])}\t{rank}\t{float(scores[pos]):.9g}\n')
    finish(folder, name, {'passages': len(pids), 'queries': len(queries),
        'indexing_seconds': build_seconds, 'mean_query_latency_ms': float(np.mean(times) * 1000),
        'timing_note': 'Single warm run; preprocessing, scoring and ranking included; loading excluded.'}, run_key)


def semantic(queries, run_key, batch_size):
    name = 'semantic_chunked'
    folder = OUT / name
    if completed(folder, run_key):
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Check PyTorch before semantic indexing.')
    shards = folder / 'embedding_batches'
    shards.mkdir(exist_ok=True)
    print('\nLoading saved dev model for test passage encoding...', flush=True)
    model = SentenceTransformer(str(MODEL), device='cuda')
    model.max_seq_length = 512
    tokenizer = model.tokenizer
    index = faiss.IndexFlatIP(384)
    pids = []
    iterator = iter(read_tsv(DATA / 'collection.tsv'))
    block_number = passages = long_passages = 0
    prep_seconds = encoding_seconds = faiss_seconds = 0.0
    with (folder / 'chunks.jsonl').open('w', encoding='utf-8') as contexts:
        with tqdm(total=TOTAL, desc='Encoding test passages') as progress:
            while True:
                rows = list(islice(iterator, 2048))
                if not rows:
                    break
                signature = hashlib.sha256(json.dumps([run_key, rows], ensure_ascii=False).encode()).hexdigest()
                checkpoint = shards / f'batch_{block_number:05d}.npz'
                if checkpoint.exists():
                    with np.load(checkpoint, allow_pickle=False) as saved:
                        if saved['signature'].item() != signature:
                            raise ValueError(f'Checkpoint mismatch: {checkpoint}')
                        vectors, ids = saved['vectors'].copy(), saved['pids'].copy()
                        texts = json.loads(saved['texts_json'].item())
                        stats = json.loads(saved['stats_json'].item())
                else:
                    start = time.perf_counter()
                    sizes = tokenizer(['passage: ' + text for _, text in rows],
                        truncation=False, padding=False, return_length=True,
                        return_attention_mask=False, return_token_type_ids=False,
                        verbose=False)['length']
                    ids, texts = [], []
                    n_long = 0
                    for (pid, text), size in zip(rows, sizes):
                        bodies = [text]
                        if size > 512:
                            bodies = split_long(text, tokenizer)
                            n_long += 1
                        ids.extend([pid] * len(bodies))
                        texts.extend(bodies)
                    prep_time = time.perf_counter() - start
                    start = time.perf_counter()
                    vectors = model.encode(['passage: ' + text for text in texts],
                        batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
                    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
                    stats = {'preparation_seconds': prep_time, 'long_passages': n_long,
                             'encoding_seconds': time.perf_counter() - start}
                    ids = np.asarray(ids, dtype=np.int64)
                    temporary = checkpoint.with_suffix('.tmp')
                    with temporary.open('wb') as file:
                        np.savez(file, signature=np.asarray(signature), vectors=vectors,
                            pids=ids, texts_json=np.asarray(json.dumps(texts, ensure_ascii=False)),
                            stats_json=np.asarray(json.dumps(stats)))
                    temporary.replace(checkpoint)
                if vectors.shape != (len(ids), 384) or len(texts) != len(ids) or not np.isfinite(vectors).all():
                    raise ValueError('Invalid embedding checkpoint.')
                start = time.perf_counter()
                offset = index.ntotal
                index.add(np.ascontiguousarray(vectors, dtype=np.float32))
                faiss_seconds += time.perf_counter() - start
                for i, (pid, text) in enumerate(zip(ids, texts)):
                    contexts.write(json.dumps({'vector_id': offset + i,
                        'passage_id': int(pid), 'text': text}, ensure_ascii=False) + '\n')
                pids.extend(ids.tolist())
                passages += len(rows)
                long_passages += stats['long_passages']
                prep_seconds += stats['preparation_seconds']
                encoding_seconds += stats['encoding_seconds']
                block_number += 1
                progress.update(len(rows))
    if passages != TOTAL or len(set(pids)) != TOTAL:
        raise ValueError('Unexpected test passage count or ID coverage.')
    pids = np.asarray(pids, dtype=np.int64)
    faiss.write_index(index, str(folder / 'index.faiss'))
    np.save(folder / 'passage_ids.npy', pids)
    model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    model.encode(['query: warmup'], normalize_embeddings=True, show_progress_bar=False)
    times = []
    with (folder / 'rankings.tsv').open('w', encoding='utf-8') as file:
        for qid, text in tqdm(queries, desc='Test semantic search'):
            start = time.perf_counter()
            query = np.ascontiguousarray(model.encode(['query: ' + text],
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False), dtype=np.float32)
            matches = search_unique(index, query, pids)
            times.append(time.perf_counter() - start)
            for rank, (pid, (score, _)) in enumerate(matches, 1):
                file.write(f'{qid}\t{pid}\t{rank}\t{score:.9g}\n')
    finish(folder, name, {'passages': passages, 'queries': len(queries),
        'vectors': index.ntotal, 'long_passages': long_passages,
        'model_relative_path': str(MODEL.relative_to(ROOT)),
        'body_window_tokens': 480, 'overlap_tokens': 64,
        'aggregation': 'maximum chunk similarity',
        'preparation_seconds': prep_seconds, 'encoding_seconds': encoding_seconds,
        'faiss_build_seconds': faiss_seconds,
        'indexing_seconds': prep_seconds + encoding_seconds + faiss_seconds,
        'timing_note': 'Preparation, encoding and FAISS add; excludes model load and file I/O. Resumed checkpoints retain original processing times.',
        'mean_query_latency_ms': float(np.mean(times) * 1000),
        'query_embedding_device': 'cpu', 'cpu_threads': 4}, run_key)


def hybrid(queries, run_key):
    # Use query IDs only, not test relevance labels, when loading fusion inputs.
    query_ids = {qid: set() for qid, _ in queries}
    runs = {key: load_rankings(OUT / folder / 'rankings.tsv', query_ids)
        for key, folder in (('tfidf', 'tfidf'), ('bm25', 'bm25'), ('semantic', 'semantic_chunked'))}
    for name, weights in (('hybrid_chunked_three', (0.25, 1, 4)),
                          ('hybrid_chunked_two', (0, 1, 4))):
        folder = OUT / name
        if completed(folder, run_key):
            continue
        start = time.perf_counter()
        scored, _ = fuse(runs, query_ids, weights)
        fusion_ms = (time.perf_counter() - start) * 1000 / len(queries)
        with (folder / 'rankings.tsv').open('w', encoding='utf-8') as file:
            for qid in sorted(scored):
                for rank, (pid, score) in enumerate(scored[qid], 1):
                    file.write(f'{qid}\t{pid}\t{rank}\t{score:.12g}\n')
        finish(folder, name, {'passages': TOTAL, 'queries': len(queries), 'rrf_k': 60,
            'weights': dict(zip(('tfidf', 'bm25', 'semantic_chunked'), weights)),
            'mean_fusion_latency_ms': fusion_ms, 'mean_query_latency_ms': None,
            'timing_note': 'Offline fusion, not end-to-end query latency.'}, run_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive.')
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    print('Verifying frozen settings and input files...', flush=True)
    run_key = initialize()
    queries = list(read_tsv(DATA / 'questions.search.tsv'))
    if len(queries) != 661 or len({qid for qid, _ in queries}) != 661:
        raise ValueError('Expected 661 unique test search queries.')
    lexical('tfidf', queries, run_key)
    gc.collect()
    lexical('bm25', queries, run_key)
    gc.collect()
    semantic(queries, run_key, args.batch_size)
    gc.collect()
    torch.cuda.empty_cache()
    hybrid(queries, run_key)
    # Relevance labels are used for evaluation only after all runs are built.
    rows = []
    for method in METHODS:
        subprocess.run([sys.executable, str(ROOT / 'evaluate_retrieval.py'),
                        '--method', method, '--split', 'test'], check=True)
        metrics = json.loads((OUT / method / 'metrics.json').read_text(encoding='utf-8'))
        row = {'method': method}
        for key in ('precision_at_10', 'success_at_5', 'recall_at_10', 'recall_at_100',
                    'map_at_100', 'ndcg_at_10', 'mean_query_latency_ms'):
            row[key] = metrics.get(key)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'comparison.csv', index=False)
    log_entry('HELD-OUT TEST EVALUATION COMPLETED', {
        'why': 'Evaluate settings frozen on dev; no tuning on test.',
        'queries': len(queries), 'comparison': rows,
        'note': 'Hybrid end-to-end timing still pending; null is not zero.'})
    print('\nFINAL TEST COMPARISON')
    print(table.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print(f'\nSaved: {OUT / "comparison.csv"}')
    print('Appended results to local processss.txt. Dev outputs preserved.')


if __name__ == '__main__':
    main()
