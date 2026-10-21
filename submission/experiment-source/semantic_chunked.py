"""Selective chunking for Lifestyle dev; reuse verified baseline checkpoints."""
import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts' / 'lifestyle' / 'dev' / 'semantic'
OUT = ROOT / 'artifacts' / 'lifestyle' / 'dev' / 'semantic_chunked'
DATA = ROOT / 'data' / 'lotte' / 'lifestyle' / 'dev'
MODEL_NAME = 'intfloat/e5-small-v2'
LIMIT, WINDOW, OVERLAP, BLOCK = 512, 480, 64, 2048


def read_tsv(path):
    with path.open(encoding='utf-8') as file:
        for line in file:
            pid, text = line.rstrip('\r\n').split('\t', 1)
            yield int(pid), text


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def split_long(text, tokenizer):
    """Retain overlapping token windows and verify actual encoded input size."""
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False,
                           verbose=False)
    start = 0
    chunks = []
    while start < len(ids):
        end = min(start + WINDOW, len(ids))
        while True:
            body = tokenizer.decode(ids[start:end], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            count = len(tokenizer.encode('passage: ' + body, truncation=False,
                                         verbose=False))
            if count <= LIMIT:
                break
            end -= max(1, count - LIMIT)
            if end <= start:
                raise ValueError('Cannot fit a chunk within the model limit.')
        chunks.append(body)
        if end == len(ids):
            break
        start = max(start + 1, end - OVERLAP)
    return chunks


def search_unique(index, query, pids, count=100):
    """Exact max-chunk passage scores with adaptive candidate expansion."""
    k = min(index.ntotal, max(count * 4, count))
    while True:
        scores, positions = index.search(query, k)
        best = {}
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            pid = int(pids[position])
            if pid not in best:
                best[pid] = (float(score), int(position))
        ranked = sorted(best.items(), key=lambda item: (-item[1][0], item[0]))
        if k == index.ntotal:
            return ranked[:count]
        # Expand through any tie at the unique-passage cutoff as well.
        if len(ranked) >= count and float(scores[0][-1]) < ranked[count - 1][1][0]:
            return ranked[:count]
        k = min(index.ntotal, k * 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not detected. Check PyTorch before proceeding.')
    model_path = BASE / 'embedding_model'
    if not (model_path / 'modules.json').exists():
        raise FileNotFoundError(f'Missing saved model: {model_path}')
    OUT.mkdir(parents=True, exist_ok=True)
    shards = OUT / 'embedding_batches'
    shards.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    print('Checking saved model and loading on GPU...', flush=True)
    model_hash = hashlib.sha256()
    for path in sorted(p for p in model_path.rglob('*') if p.is_file()):
        model_hash.update(str(path.relative_to(model_path)).encode())
        model_hash.update(hash_file(path).encode())
    model_digest = model_hash.hexdigest()
    model = SentenceTransformer(str(model_path), device='cuda')
    model.max_seq_length = LIMIT
    tokenizer = model.tokenizer
    index = faiss.IndexFlatIP(384)
    passage_map = []
    reused = long_count = new_count = 0
    encoding_seconds = faiss_seconds = 0.0
    block_number = 0
    iterator = iter(read_tsv(DATA / 'collection.tsv'))
    # Context text is saved separately for the future RAG interface.
    with (OUT / 'chunks.jsonl').open('w', encoding='utf-8') as contexts:
        with tqdm(total=268893, desc='Reusing / chunking passages') as progress:
            while True:
                rows = list(islice(iterator, BLOCK))
                if not rows:
                    break
                source = BASE / 'embedding_batches' / f'batch_{block_number:05d}.npz'
                signature = hashlib.sha256(json.dumps(
                    [MODEL_NAME, LIMIT, rows], ensure_ascii=False).encode()).hexdigest()
                with np.load(source, allow_pickle=False) as old:
                    if old['signature'].item() != signature:
                        raise ValueError(f'Baseline text/settings mismatch: {source}')
                    if not np.array_equal(old['pids'], [pid for pid, _ in rows]):
                        raise ValueError(f'Baseline PID mismatch: {source}')
                    old_vectors = old['vectors'].copy()
                if old_vectors.shape != (len(rows), 384):
                    raise ValueError('Unexpected baseline embedding shape.')
                key = hashlib.sha256(json.dumps([
                    signature, hash_file(source), model_digest, WINDOW, OVERLAP,
                    'decode-checked-v1']).encode()).hexdigest()
                checkpoint = shards / f'batch_{block_number:05d}.npz'
                if checkpoint.exists():
                    with np.load(checkpoint, allow_pickle=False) as saved:
                        if saved['key'].item() != key:
                            raise ValueError(f'Chunk checkpoint mismatch: {checkpoint}')
                        vectors = saved['vectors'].copy()
                        ids = saved['pids'].copy()
                        texts = json.loads(saved['texts_json'].item())
                        stats = json.loads(saved['stats_json'].item())
                else:
                    sizes = tokenizer(
                        ['passage: ' + text for _, text in rows],
                        truncation=False, padding=False, return_length=True,
                        return_attention_mask=False, return_token_type_ids=False,
                        verbose=False)['length']
                    ids, texts, vectors_list, pending, slots = [], [], [], [], []
                    short_n = long_n = 0
                    for row_i, ((pid, text), size) in enumerate(zip(rows, sizes)):
                        if size <= LIMIT:
                            ids.append(pid)
                            texts.append(text)
                            vectors_list.append(old_vectors[row_i])
                            short_n += 1
                        else:
                            long_n += 1
                            for body in split_long(text, tokenizer):
                                slots.append(len(ids))
                                ids.append(pid)
                                texts.append(body)
                                vectors_list.append(None)
                                pending.append('passage: ' + body)
                    start = time.perf_counter()
                    if pending:
                        encoded = model.encode(
                            pending, batch_size=args.batch_size,
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False)
                        for slot, vector in zip(slots, encoded):
                            vectors_list[slot] = vector
                    elapsed = time.perf_counter() - start
                    vectors = np.asarray(vectors_list, dtype=np.float32)
                    ids = np.asarray(ids, dtype=np.int64)
                    stats = {'reused': short_n, 'long': long_n,
                             'new': len(pending), 'encoding_seconds': elapsed}
                    temporary = checkpoint.with_suffix('.tmp')
                    with temporary.open('wb') as file:
                        np.savez(file, vectors=vectors, pids=ids,
                                 key=np.asarray(key),
                                 texts_json=np.asarray(json.dumps(texts, ensure_ascii=False)),
                                 stats_json=np.asarray(json.dumps(stats)))
                    temporary.replace(checkpoint)
                if vectors.shape != (len(ids), 384) or len(texts) != len(ids):
                    raise ValueError('Chunk checkpoint shape mismatch.')
                if not np.isfinite(vectors).all():
                    raise ValueError('Invalid chunk vectors.')
                start = time.perf_counter()
                offset = index.ntotal
                index.add(np.ascontiguousarray(vectors, dtype=np.float32))
                faiss_seconds += time.perf_counter() - start
                passage_map.extend(ids.tolist())
                for i, (pid, body) in enumerate(zip(ids, texts)):
                    contexts.write(json.dumps({'vector_id': offset + i,
                        'passage_id': int(pid), 'text': body}, ensure_ascii=False) + '\n')
                reused += stats['reused']
                long_count += stats['long']
                new_count += stats['new']
                encoding_seconds += stats['encoding_seconds']
                block_number += 1
                progress.update(len(rows))
    pids = np.asarray(passage_map, dtype=np.int64)
    if len(set(passage_map)) != reused + long_count:
        raise ValueError('Original passage count mismatch.')
    faiss.write_index(index, str(OUT / 'index.faiss'))
    np.save(OUT / 'passage_ids.npy', pids)
    print(f'Reused: {reused:,}; long passages: {long_count:,}; new chunks: {new_count:,}', flush=True)
    model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    model.encode(['query: warmup'], normalize_embeddings=True, show_progress_bar=False)
    queries = list(read_tsv(DATA / 'questions.search.tsv'))
    latencies = []
    with (OUT / 'rankings.tsv').open('w', encoding='utf-8') as file:
        for qid, text in tqdm(queries, desc='Evaluating chunked search'):
            start = time.perf_counter()
            query = np.ascontiguousarray(model.encode(
                ['query: ' + text], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
            matches = search_unique(index, query, pids)
            latencies.append(time.perf_counter() - start)
            for rank, (pid, (score, _)) in enumerate(matches, 1):
                file.write(f'{qid}\t{pid}\t{rank}\t{score:.9g}\n')
    metrics = {
        'method': 'semantic_chunked', 'domain': 'lifestyle', 'split': 'dev',
        'query_type': 'search', 'model': MODEL_NAME, 'model_sha256': model_digest,
        'model_relative_path': str(model_path.relative_to(ROOT)),
        'passages': reused + long_count, 'queries': len(queries),
        'vectors': index.ntotal, 'reused_short_passage_embeddings': reused,
        'long_passages': long_count, 'new_chunk_embeddings': new_count,
        'body_window_tokens': WINDOW, 'overlap_tokens': OVERLAP,
        'max_input_tokens': LIMIT, 'aggregation': 'maximum chunk similarity',
        'candidate_limit': 100, 'query_embedding_device': 'cpu', 'cpu_threads': 4,
        'additional_encoding_seconds': encoding_seconds,
        'faiss_build_seconds': faiss_seconds,
        'indexing_seconds': None,
        'timing_note': 'Incremental build reuses baseline embeddings; encoding times include resumed batches. Not a from-scratch build time.',
        'mean_query_latency_ms': float(np.mean(latencies) * 1000),
    }
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('\nCHUNKED SEMANTIC RETRIEVAL COMPLETE')
    print(json.dumps(metrics, indent=2))
    stamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d %H:%M IST')
    with (ROOT / 'processss.txt').open('a', encoding='utf-8') as log:
        log.write(f'\n\nCHUNKED SEMANTIC BUILD | {stamp}\n')
        log.write('What: Reused short passage vectors; encoded overlapping long-passage chunks.\n')
        log.write('Why: Avoid long-passage truncation; retain original passage IDs for scoring.\n')
        log.write(json.dumps(metrics, indent=2) + '\n')
        log.write('Quality evaluation pending. Original semantic index preserved.\n')
    print(f'\nSaved to: {OUT}')
    print('Next: python evaluate_retrieval.py --method semantic_chunked --split dev')


if __name__ == '__main__':
    main()
