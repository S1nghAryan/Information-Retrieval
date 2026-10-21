"""Bounded local validation; does not rebuild indexes or change experiments."""
import json
from pathlib import Path
import requests

OUT=Path(__file__).resolve().parent.parent/'submission'
BASE='http://127.0.0.1:5173/api/'
cases=[
 ('plants','Which plants grow well indoors?','hybrid'),
 ('hiking','What should I pack for a day hike?','hybrid_chunked_two'),
 ('bread','How should I store homemade bread?','bm25'),
 ('out_of_scope','What is the exact weather forecast for Mars tomorrow?','hybrid_chunked_two'),
 ('no_matches','zzqvxxnonsenseunmatchedword987654321','tfidf'),
]
records=[]
for name,question,method in cases:
    r=requests.post(BASE+'search',json={'question':question,'method':method},timeout=180)
    r.raise_for_status(); search=r.json()
    ids=[p['passage_id'] for p in search['passages']]
    assert len(ids)==len(set(ids)) and len(ids)<=10
    r=requests.post(BASE+'answer',json={'retrieval_id':search['retrieval_id']},timeout=200)
    r.raise_for_status(); answer=r.json()
    assert answer['answer'] and '<think>' not in answer['answer'] and '</think>' not in answer['answer']
    assert all(c in [s['rank'] for s in answer['context_sources']] for c in answer['citations'])
    assert not answer['warnings'], answer['warnings']
    for quote in answer.get('evidence_quotes', []):
        source = next(s for s in answer['context_sources'] if s['rank'] == quote['source'])
        assert quote['quote'] in source['text']
    if name == 'bread':
        assert answer.get('answer_status') == 'quoted_evidence'
        assert 'homemade' not in answer['answer'].lower()
    if name == 'out_of_scope':
        assert answer.get('answer_status') == 'insufficient_evidence'
    records.append({'case':name,'question':question,'method':method,'search':search,'answer':answer})
    (OUT/'answer-validation.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps({'case':name,'answer':answer['answer'],'context':answer['context_sources'],'warnings':answer['warnings']},ensure_ascii=True),flush=True)
checks=[]
for body in [{'question':' ','method':'tfidf'},{'question':'a'*501,'method':'bm25'},{'question':'hello','method':'unknown'}]:
    r=requests.post(BASE+'search',json=body,timeout=10)
    assert r.status_code==422
    checks.append({'check':'invalid search input','status':r.status_code})
r=requests.post(BASE+'answer',json={'retrieval_id':'missing'},timeout=10)
assert r.status_code==410
checks.append({'check':'expired or missing result','status':410})
(OUT/'input-validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('Input validation passed.',flush=True)
