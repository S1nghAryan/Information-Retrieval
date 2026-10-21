"""Local RAG API. Existing experiment files are opened read-only."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json
import re
import time
import uuid
import threading
from collections import OrderedDict
from typing import Literal
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from generation import generate

ROOT = Path(os.environ.get('RAG_ROOT', str(Path(__file__).resolve().parents[1])))
BASE = ROOT / 'artifacts/lifestyle/dev'
OLLAMA = 'http://127.0.0.1:11434'
MODEL = 'qwen3:4b'
METHODS = Literal['tfidf', 'bm25', 'semantic_chunked', 'hybrid_chunked_two', 'hybrid']
app = FastAPI(title='Fieldnotes local RAG')
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:5173','http://localhost:5173'], allow_methods=['GET','POST'], allow_headers=['Content-Type'])
lock = threading.Lock()
generation_lock = threading.Lock()
engine = None
cache = OrderedDict()

class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    method: METHODS = 'hybrid_chunked_two'

class AnswerRequest(BaseModel):
    retrieval_id: str

def ranked_scores(scores, pids):
    import numpy as np
    candidates = np.flatnonzero(scores > 0)
    order = np.lexsort((pids[candidates], -scores[candidates]))
    return [(int(pids[i]), float(scores[i])) for i in candidates[order[:100]]]

def fuse(runs, weights):
    scores = {}
    for run, weight in zip(runs, weights):
        for rank, (pid, _) in enumerate(run, 1):
            scores[pid] = scores.get(pid, 0.0) + weight / (60 + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:100]

class Engine:
    def __init__(self):
        import numpy as np
        import faiss
        import torch
        import Stemmer
        from sentence_transformers import SentenceTransformer
        # Import only the existing exact max-chunk search function; main is not run.
        sys.path.append(str(ROOT))
        from semantic_chunked import search_unique
        self.search_unique = search_unique
        torch.set_num_threads(4)
        faiss.omp_set_num_threads(4)
        self.stemmer = Stemmer.Stemmer('english')
        self.model = SentenceTransformer(str(BASE / 'semantic/embedding_model'), device='cpu', local_files_only=True)
        self.model.max_seq_length = 512
        self.index = faiss.read_index(str(BASE / 'semantic_chunked/index.faiss'))
        self.pids = np.load(BASE / 'semantic_chunked/passage_ids.npy', allow_pickle=False)
        if len(self.pids) != self.index.ntotal:
            raise ValueError('Saved semantic index and passage mapping do not match.')
        self.lexical = {}
        self.original_semantic = None
        self.chunks_path = BASE / 'semantic_chunked/chunks.jsonl'
        self.chunk_offsets = []
        with self.chunks_path.open('rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                self.chunk_offsets.append(offset)
        if len(self.chunk_offsets) != self.index.ntotal:
            raise ValueError('Saved chunk text does not match the vector count.')
        self.collection = ROOT / 'data/lotte/lifestyle/dev/collection.tsv'
        self.passage_offsets = {}
        with self.collection.open('rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                self.passage_offsets[int(line.split(b'\t', 1)[0])] = offset
        self.model.encode(['query: warmup'], normalize_embeddings=True, show_progress_bar=False)

    def load_lexical(self, name):
        if name in self.lexical:
            return
        import numpy as np
        folder = BASE / name
        pids = np.load(folder / 'passage_ids.npy', allow_pickle=False)
        if name == 'tfidf':
            import joblib
            from scipy.sparse import load_npz
            self.lexical[name] = (joblib.load(folder / 'vectorizer.joblib'), load_npz(folder / 'matrix.npz').T.tocsr(), pids)
        else:
            import bm25s
            self.lexical[name] = (bm25s.BM25.load(str(folder / 'index'), mmap=True), pids)

    def search(self, question, method):
        import numpy as np
        needed = ['tfidf', 'bm25'] if method == 'hybrid' else ['bm25'] if method == 'hybrid_chunked_two' else [method] if method in ('tfidf','bm25') else []
        load_start = time.perf_counter()
        for name in needed:
            self.load_lexical(name)
        if method == 'hybrid' and self.original_semantic is None:
            import faiss
            self.original_semantic = (faiss.read_index(str(BASE / 'semantic/index.faiss')), np.load(BASE / 'semantic/passage_ids.npy', allow_pickle=False))
        load_ms = (time.perf_counter()-load_start)*1000
        start = time.perf_counter()
        processed = ' '.join(self.stemmer.stemWords(re.findall(r'\b\w+\b', question.lower())))
        runs = {}
        for name in needed:
            if name == 'tfidf':
                vectorizer, postings, ids = self.lexical[name]
                scores = (vectorizer.transform([processed]) @ postings).toarray().ravel()
            else:
                retriever, ids = self.lexical[name]
                tokens = [t for t in processed.split() if t in retriever.vocab_dict]
                scores = retriever.get_scores(tokens) if tokens else np.zeros(len(ids))
            runs[name] = ranked_scores(scores, ids)
        positions = {}
        if method not in ('tfidf','bm25'):
            query = np.ascontiguousarray(self.model.encode(['query: ' + question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
            if method == 'hybrid':
                original_index, original_pids = self.original_semantic
                scores, indices = original_index.search(query, 100)
                runs['semantic'] = sorted([(int(original_pids[i]), float(score)) for i, score in zip(indices[0], scores[0]) if i >= 0], key=lambda pair: (-pair[1], pair[0]))
            else:
                matches = self.search_unique(self.index, query, self.pids)
                runs['semantic_chunked'] = [(pid, score) for pid, (score, _) in matches]
                positions = {pid: position for pid, (_, position) in matches}
        if method == 'hybrid':
            ranked = fuse([runs['tfidf'], runs['bm25'], runs['semantic']], [1, 1, 1])
        elif method == 'hybrid_chunked_two':
            ranked = fuse([runs['bm25'], runs['semantic_chunked']], [1, 4])
        else:
            ranked = runs[method]
        retrieval_ms = (time.perf_counter()-start)*1000
        passages = []
        with self.collection.open('rb') as collection, self.chunks_path.open('rb') as chunks:
            for rank, (pid, score) in enumerate(ranked[:10], 1):
                collection.seek(self.passage_offsets[pid])
                text = collection.readline().decode('utf-8').rstrip('\r\n').split('\t', 1)[1]
                context = text
                if pid in positions:
                    position = positions[pid]
                    chunks.seek(self.chunk_offsets[position])
                    chunk = json.loads(chunks.readline())
                    if chunk['vector_id'] != position or chunk['passage_id'] != pid:
                        raise ValueError('Saved chunk mapping mismatch.')
                    context = chunk['text']
                passages.append({'rank': rank, 'passage_id': pid, 'score': score, 'text': text, 'context': context, 'context_type': 'Best matching chunk' if pid in positions else 'Original passage'})
        return passages, retrieval_ms, load_ms

@app.get('/api/health')
def health():
    try:
        r = requests.get(OLLAMA + '/api/tags', timeout=2)
        r.raise_for_status()
        models = [m['name'] for m in r.json().get('models', [])]
        available = MODEL in models
    except requests.RequestException:
        available = False
    return {'status': 'ok', 'retrieval_loaded': engine is not None, 'ollama_available': available, 'model': MODEL, 'collection': 'Lifestyle dev', 'passages': 268893}

@app.post('/api/search')
def search(body: SearchRequest):
    global engine
    question = body.question.strip()
    if not question or len(question.encode('utf-8')) > 1200:
        raise HTTPException(422, 'Enter a question of at most 500 characters / 1,200 UTF-8 bytes.')
    with lock:
        started = time.perf_counter()
        try:
            if engine is None:
                engine = Engine()
            cold_ms = (time.perf_counter()-started)*1000
            passages, retrieval_ms, lexical_load_ms = engine.search(question, body.method)
        except Exception as exc:
            print(f'Retrieval error: {exc}', flush=True)
            raise HTTPException(503, 'Could not load or search the saved indexes. Check the backend log.') from exc
        result_id = str(uuid.uuid4())
        result = {'retrieval_id': result_id, 'question': question, 'method': body.method, 'collection': 'Lifestyle dev', 'passages': passages, 'retrieval_ms': retrieval_ms, 'index_load_ms': cold_ms + lexical_load_ms, 'source_lookup_included_in_retrieval_timing': False}
        cache[result_id] = (time.monotonic(), result)
        while len(cache) > 32:
            cache.popitem(last=False)
        return result

def clipped(text, budget):
    if len(text.encode('utf-8')) <= budget:
        return text
    fragment = text.encode('utf-8')[:budget].decode('utf-8', errors='ignore')
    # Do not expose an unfinished instruction as though it were complete evidence.
    endings = list(re.finditer(r'[.!?](?:\s|$)', fragment))
    if endings:
        return fragment[:endings[-1].start()+1]
    return fragment.rsplit(' ', 1)[0] + ' [excerpt truncated; sentence incomplete]'

@app.post('/api/answer')
def answer(body: AnswerRequest):
    with lock:
        item = cache.get(body.retrieval_id)
    if item is None or time.monotonic() - item[0] > 1800:
        raise HTTPException(410, 'Search results expired. Run your question again.')
    result = item[1]
    sources = result['passages'][:10]
    if not sources:
        return {'answer': 'No matching passages were found. Try another question or retrieval method.', 'generation_ms': 0, 'model': MODEL, 'citations': [], 'context_sources': [], 'warnings': []}
    # Ten bounded excerpts leave room for instructions and output in the 8192-token window.
    contexts = [{'rank': p['rank'], 'passage_id': p['passage_id'], 'text': clipped(p['context'], 550)} for p in sources]
    started = time.perf_counter()
    if not generation_lock.acquire(blocking=False):
        raise HTTPException(429, 'Another answer is being generated. Please retry shortly.')
    try:
        generated = generate(result['question'], contexts, requests.post, OLLAMA, MODEL)
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(503, 'Ollama could not generate an answer. Your retrieved sources are still available. Check that qwen3:4b is running, then retry.') from exc
    finally:
        generation_lock.release()
    return {**generated, 'generation_ms': (time.perf_counter()-started)*1000, 'model': MODEL, 'context_sources': contexts}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
