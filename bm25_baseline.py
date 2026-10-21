import gc
import json
import time
from importlib.metadata import version
from pathlib import Path

import bm25s
import numpy as np
from tqdm import tqdm

# Reuse exactly the same preprocessing as TF-IDF.
# Importing this file does not run its main() function.
from tfidf_baseline import preprocess, read_tsv


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "lotte" / "lifestyle" / "dev"
OUTPUT = ROOT / "artifacts" / "lifestyle" / "dev" / "bm25"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    passage_ids = []
    processed_texts = []

    print("Preparing BM25 passages...", flush=True)
    start = time.perf_counter()

    for pid, text in tqdm(
        read_tsv(DATA / "collection.tsv"),
        total=268893,
        desc="Preprocessing passages",
    ):
        passage_ids.append(pid)
        processed_texts.append(preprocess(text))

    # Text is already lowercased and stemmed.
    # Keep one-character tokens and do not remove stopwords,
    # matching the TF-IDF baseline.
    corpus_tokens = bm25s.tokenize(
        processed_texts,
        stopwords=[],
        stemmer=None,
        token_pattern=r"(?u)\b\w+\b",
    )

    del processed_texts
    gc.collect()

    print("Building BM25 index...", flush=True)
    retriever = bm25s.BM25(
        k1=1.5,
        b=0.75,
        method="lucene",
    )
    retriever.index(corpus_tokens)

    indexing_seconds = time.perf_counter() - start
    del corpus_tokens
    gc.collect()

    pids = np.asarray(passage_ids, dtype=np.int64)
    del passage_ids

    # Store the index and its original passage-ID mapping.
    retriever.save(str(OUTPUT / "index"))
    np.save(OUTPUT / "passage_ids.npy", pids)

    print(f"Passages indexed: {len(pids):,}")
    print(f"Indexing time: {indexing_seconds:.1f} seconds")

    queries = list(read_tsv(DATA / "questions.search.tsv"))
    latencies = []

    with (OUTPUT / "rankings.tsv").open(
        "w", encoding="utf-8"
    ) as rankings:
        for qid, question in tqdm(queries, desc="Retrieving queries"):
            start = time.perf_counter()

            tokens = preprocess(question).split()

            # Ignore query terms absent from the indexed vocabulary.
            tokens = [
                token for token in tokens
                if token in retriever.vocab_dict
            ]

            if tokens:
                scores = retriever.get_scores(tokens)
                candidates = np.flatnonzero(scores > 0)

                # Same tie-breaking rule as TF-IDF:
                # descending score, then ascending original passage ID.
                order = np.lexsort(
                    (pids[candidates], -scores[candidates])
                )
                top = candidates[order[:100]]
            else:
                scores = np.zeros(len(pids), dtype=np.float32)
                top = np.array([], dtype=np.int64)

            latencies.append(time.perf_counter() - start)

            for rank, position in enumerate(top, start=1):
                rankings.write(
                    f"{qid}\t{int(pids[position])}\t{rank}\t"
                    f"{float(scores[position]):.9g}\n"
                )

    metrics = {
        "method": "bm25",
        "domain": "lifestyle",
        "split": "dev",
        "query_type": "search",
        "passages": len(pids),
        "queries": len(queries),
        "candidate_limit": 100,
        "bm25_variant": "lucene",
        "k1": 1.5,
        "b": 0.75,
        "bm25s_version": version("bm25s"),
        "preprocessing": "lowercase, word tokens, English stemming",
        "stopword_removal": False,
        "mean_query_latency_ms": float(np.mean(latencies) * 1000),
        "indexing_seconds": indexing_seconds,
    }

    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\nBM25 retrieval complete")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved files: {OUTPUT}")


if __name__ == "__main__":
    main()