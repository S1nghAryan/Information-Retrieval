"""Audit dev passage lengths without changing retrieval indexes."""
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'lotte' / 'lifestyle' / 'dev'
MODEL = ROOT / 'artifacts' / 'lifestyle' / 'dev' / 'semantic' / 'embedding_model'
OUT = ROOT / 'artifacts' / 'lifestyle' / 'dev' / 'length_audit'
LIMIT = 512
# Planning estimates only: exact chunk construction is a later step.
BODY_TOKENS = 480
OVERLAP = 64


def records():
    with (DATA / 'collection.tsv').open(encoding='utf-8') as file:
        for line in file:
            pid, text = line.rstrip('\r\n').split('\t', 1)
            yield int(pid), text


def main():
    if not (MODEL / 'tokenizer_config.json').exists():
        raise FileNotFoundError(f'Saved tokenizer not found in {MODEL}')
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    OUT.mkdir(parents=True, exist_ok=True)
    iterator = iter(records())
    lengths, long_ids = [], set()
    estimated_chunks = 0
    total_body_tokens = 0
    start_time = datetime.now(timezone(timedelta(hours=5, minutes=30)))

    with (OUT / 'long_passages.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['passage_id', 'prefixed_input_tokens', 'body_tokens', 'estimated_chunks'])
        with tqdm(total=268893, desc='Counting tokens (CPU)') as progress:
            while True:
                batch = list(islice(iterator, 256))
                if not batch:
                    break
                texts = [text for _, text in batch]
                # Count the exact input used by our semantic baseline.
                encoded = tokenizer(
                    ['passage: ' + text for text in texts],
                    add_special_tokens=True, truncation=False, padding=False,
                    return_length=True, return_attention_mask=False,
                    return_token_type_ids=False, verbose=False,
                )
                input_lengths = encoded['length']
                del encoded
                bodies = tokenizer(
                    texts, add_special_tokens=False, truncation=False,
                    padding=False, return_length=True,
                    return_attention_mask=False, return_token_type_ids=False,
                    verbose=False,
                )['length']
                for (pid, _), size, body in zip(batch, input_lengths, bodies):
                    size, body = int(size), int(body)
                    lengths.append(size)
                    total_body_tokens += body
                    chunks = 1
                    if size > LIMIT:
                        long_ids.add(pid)
                        chunks = 1 + math.ceil(max(0, body - BODY_TOKENS) / (BODY_TOKENS - OVERLAP))
                        writer.writerow([pid, size, body, chunks])
                    estimated_chunks += chunks
                progress.update(len(batch))

    lengths = np.asarray(lengths, dtype=np.int64)
    if not len(lengths):
        raise ValueError('Collection is empty.')
    # Diagnostic only: labels do not affect chunk sizes or indexing.
    relevant_pairs = affected_pairs = affected_queries = query_count = 0
    with (DATA / 'qas.search.jsonl').open(encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            relevant = {int(pid) for pid in item['answer_pids']}
            affected = len(relevant & long_ids)
            relevant_pairs += len(relevant)
            affected_pairs += affected
            affected_queries += int(affected > 0)
            query_count += 1

    report = {
        'domain': 'lifestyle', 'split': 'dev', 'passages': len(lengths),
        'input_count_includes': 'passage prefix and special tokens',
        'model_limit': LIMIT,
        'passages_over_limit': len(long_ids),
        'percent_over_limit': 100 * len(long_ids) / len(lengths),
        'mean_input_tokens': float(lengths.mean()),
        'median_input_tokens': float(np.median(lengths)),
        'p95_input_tokens': float(np.percentile(lengths, 95)),
        'p99_input_tokens': float(np.percentile(lengths, 99)),
        'maximum_input_tokens': int(lengths.max()),
        'total_body_tokens': total_body_tokens,
        'dev_queries': query_count,
        'queries_with_at_least_one_long_relevant_passage': affected_queries,
        'labeled_relevant_pairs': relevant_pairs,
        'labeled_relevant_pairs_in_long_passages': affected_pairs,
        'proposed_body_tokens': BODY_TOKENS,
        'proposed_overlap_tokens': OVERLAP,
        'estimated_vectors_if_only_long_passages_are_split': estimated_chunks,
        'estimated_vector_count_multiplier': estimated_chunks / len(lengths),
        'estimate_note': 'Token windows only; no chunks or embeddings generated.',
    }
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nPASSAGE LENGTH AUDIT')
    print(json.dumps(report, indent=2))
    print(f'\nSaved to: {OUT}')
    with (ROOT / 'processss.txt').open('a', encoding='utf-8') as log:
        log.write('\n\nPASSAGE LENGTH AUDIT | ' + start_time.strftime('%Y-%m-%d %H:%M IST') + '\n')
        log.write('What: Counted original dev inputs with the saved E5 tokenizer.\n')
        log.write('Why: Quantify truncation before implementing required chunking.\n')
        log.write(json.dumps(report, indent=2) + '\n')
        log.write('Status: Audit only. Indexes and embeddings unchanged. Test not inspected.\n')
    print('Appended audit results to local processss.txt.')


if __name__ == '__main__':
    main()
