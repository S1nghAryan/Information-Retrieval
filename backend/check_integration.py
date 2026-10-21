"""Read-only parity check against saved evaluation rankings, plus live generation."""
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/lifestyle/dev'
with (ROOT / 'data/lotte/lifestyle/dev/questions.search.tsv').open(encoding='utf-8') as f:
    qid, question = f.readline().rstrip('\r\n').split('\t', 1)

for method in ['tfidf', 'bm25', 'semantic_chunked', 'hybrid_chunked_two', 'hybrid']:
    expected = []
    with (BASE / method / 'rankings.tsv').open(encoding='utf-8') as f:
        for line in f:
            query_id, pid, rank, score = line.split()
            if query_id == qid and int(rank) <= 10:
                expected.append((int(rank), int(pid)))
    r = requests.post('http://127.0.0.1:8000/api/search', json={'question': question, 'method': method}, timeout=120)
    r.raise_for_status()
    actual = [p['passage_id'] for p in r.json()['passages']]
    assert actual == [pid for _, pid in sorted(expected)], (method, actual, expected)
    print(f'{method}: top 10 exactly match saved rankings; {r.json()["retrieval_ms"]:.1f} ms', flush=True)

invalid = requests.post('http://127.0.0.1:8000/api/search', json={'question': ' ', 'method': 'tfidf'}, timeout=10)
assert invalid.status_code == 422
missing = requests.post('http://127.0.0.1:8000/api/answer', json={'retrieval_id': 'missing'}, timeout=10)
assert missing.status_code == 410
print('Blank query and expired result validation passed.', flush=True)

search = requests.post('http://127.0.0.1:5173/api/search', json={'question':'Which plants grow well indoors?', 'method':'hybrid'}, timeout=120)
search.raise_for_status()
saved = search.json()
r = requests.post('http://127.0.0.1:8000/api/answer', json={'retrieval_id': saved['retrieval_id']}, timeout=200)
r.raise_for_status()
answer = r.json()
assert answer['answer'] and answer['model'] == 'qwen3:4b'
assert answer['context_sources']
assert '<think>' not in answer['answer'] and '</think>' not in answer['answer']
assert answer['citations'], 'No citations generated'
assert not answer['warnings'], answer['warnings']
assert all(n in [s['rank'] for s in answer['context_sources']] for n in answer['citations'])
(Path(__file__).parent / 'smoke-answer.json').write_text(json.dumps(answer, indent=2), encoding='utf-8')
print(json.dumps({k:answer[k] for k in ['answer','generation_ms','citations','warnings']}, indent=2), flush=True)

for removed in ['semantic', 'hybrid_chunked_three', 'hybrid_reranked']:
    response = requests.post('http://127.0.0.1:8000/api/search', json={'question':'plants', 'method':removed}, timeout=10)
    assert response.status_code == 422, (removed, response.status_code)
print('Removed methods rejected by API.', flush=True)
