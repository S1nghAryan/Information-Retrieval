import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import Stemmer
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "lotte" / "lifestyle" / "dev"
OUTPUT = ROOT / "artifacts" / "lifestyle" / "dev" / "tfidf"
OUTPUT.mkdir(parents=True, exist_ok=True)

stemmer = Stemmer.Stemmer("english")


def preprocess(text):
    # Use identical preprocessing for passages and queries.
    tokens = re.findall(r"\b\w+\b", text.lower())
    return " ".join(stemmer.stemWords(tokens))


def read_tsv(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            record_id, text = line.rstrip("\r\n").split("\t", 1)
            yield int(record_id), text


def main():
    passage_ids = []

    def documents():
        # Stream passage text instead of keeping all raw text in RAM.
        for pid, text in tqdm(
            read_tsv(DATA / "collection.tsv"),
            total=268893,
            desc="Preprocessing passages",
        ):
            passage_ids.append(pid)
            yield preprocess(text)

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        norm="l2",
        sublinear_tf=True,
        smooth_idf=True,
        dtype=np.float32,
    )

    print("Building TF-IDF index...", flush=True)
    start = time.perf_counter()
    matrix = vectorizer.fit_transform(documents())
    indexing_seconds = time.perf_counter() - start
    pids = np.asarray(passage_ids, dtype=np.int64)

    print(f"Index shape: {matrix.shape}", flush=True)
    print(f"Indexing time: {indexing_seconds:.1f} seconds", flush=True)

    # These files allow us to reuse the index later.
    save_npz(OUTPUT / "matrix.npz", matrix)
    np.save(OUTPUT / "passage_ids.npy", pids)
    joblib.dump(vectorizer, OUTPUT / "vectorizer.joblib")

    # Transposed CSR matrix acts as term-to-passage postings.
    postings = matrix.T.tocsr()
    del matrix

    queries = list(read_tsv(DATA / "questions.search.tsv"))
    precisions = []
    latencies = []

    # Labels are used only to score results, never to retrieve them.
    with (DATA / "qas.search.jsonl").open(encoding="utf-8") as file:
        qrels = {
            int(item["qid"]): {int(pid) for pid in item["answer_pids"]}
            for item in (json.loads(line) for line in file)
        }

    ranking_path = OUTPUT / "rankings.tsv"
    with ranking_path.open("w", encoding="utf-8") as rankings:
        for qid, question in tqdm(queries, desc="Evaluating queries"):
            start = time.perf_counter()
            query_vector = vectorizer.transform([preprocess(question)])
            scores = (query_vector @ postings).toarray().ravel()

            # Retrieve positive matches only. Break score ties by passage ID.
            candidates = np.flatnonzero(scores > 0)
            order = np.lexsort((pids[candidates], -scores[candidates]))
            top = candidates[order[:100]]
            latencies.append(time.perf_counter() - start)

            retrieved = pids[top].tolist()
            relevant = qrels[qid]
            hits = sum(pid in relevant for pid in retrieved[:10])
            precisions.append(hits / 10)

            for rank, position in enumerate(top, start=1):
                rankings.write(
                    f"{qid}\t{int(pids[position])}\t{rank}\t"
                    f"{float(scores[position]):.9g}\n"
                )

    metrics = {
        "method": "tfidf",
        "domain": "lifestyle",
        "split": "dev",
        "query_type": "search",
        "passages": len(pids),
        "queries": len(queries),
        "candidate_limit": 100,
        "precision_at_10": float(np.mean(precisions)),
        "mean_query_latency_ms": float(np.mean(latencies) * 1000),
        "indexing_seconds": indexing_seconds,
    }

    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\nTF-IDF baseline complete")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved files: {OUTPUT}")


if __name__ == "__main__":
    main()