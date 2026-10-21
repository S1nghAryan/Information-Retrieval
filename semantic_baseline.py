import gc
import hashlib
import json
import time
from itertools import islice
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from tfidf_baseline import read_tsv


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "lotte" / "lifestyle" / "dev"
OUTPUT = ROOT / "artifacts" / "lifestyle" / "dev" / "semantic"

MODEL_NAME = "intfloat/e5-small-v2"
BATCH_SIZE = 16
CHECKPOINT_SIZE = 2048
DIMENSIONS = 384
MAX_LENGTH = 512


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Stop and check PyTorch.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoints = OUTPUT / "embedding_batches"
    checkpoints.mkdir(exist_ok=True)

    # Limit CPU threads for predictable query-time measurements.
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)

    # Save the model locally so resumed batches use the same model.
    model_path = OUTPUT / "embedding_model"
    if (model_path / "modules.json").exists():
        source = str(model_path)
    else:
        source = MODEL_NAME

    print("Loading embedding model...", flush=True)
    model = SentenceTransformer(source, device="cuda")
    model.max_seq_length = MAX_LENGTH

    if source == MODEL_NAME:
        model.save(str(model_path))

    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    index = faiss.IndexFlatIP(DIMENSIONS)
    all_pids = []
    embedding_seconds = 0.0
    faiss_build_seconds = 0.0

    records = iter(read_tsv(DATA / "collection.tsv"))
    batch_number = 0

    with tqdm(total=268893, desc="Embedding passages") as progress:
        while True:
            rows = list(islice(records, CHECKPOINT_SIZE))
            if not rows:
                break

            ids = np.asarray([pid for pid, _ in rows], dtype=np.int64)
            texts = ["passage: " + text for _, text in rows]

            # Detect accidental changes to the source passages/settings.
            signature = hashlib.sha256(
                json.dumps(
                    [MODEL_NAME, MAX_LENGTH, rows],
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()

            checkpoint = checkpoints / f"batch_{batch_number:05d}.npz"

            if checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as saved:
                    if saved["signature"].item() != signature:
                        raise ValueError(
                            f"Checkpoint mismatch: {checkpoint}. "
                            "Do not reuse checkpoints with changed input."
                        )
                    vectors = saved["vectors"].copy()
                    elapsed = float(saved["seconds"].item())
                    if not np.array_equal(saved["pids"], ids):
                        raise ValueError("Checkpoint passage IDs do not match.")
            else:
                start = time.perf_counter()
                vectors = model.encode(
                    texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                vectors = np.ascontiguousarray(vectors, dtype=np.float32)
                elapsed = time.perf_counter() - start

                # Replace only after a checkpoint has been fully written.
                temporary = checkpoint.with_suffix(".tmp")
                with temporary.open("wb") as file:
                    np.savez(
                        file,
                        vectors=vectors,
                        pids=ids,
                        signature=np.asarray(signature),
                        seconds=np.asarray(elapsed),
                    )
                temporary.replace(checkpoint)

            if vectors.shape != (len(rows), DIMENSIONS):
                raise ValueError("Unexpected embedding dimensions.")
            if not np.isfinite(vectors).all():
                raise ValueError("Embeddings contain invalid values.")

            embedding_seconds += elapsed

            start = time.perf_counter()
            index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            faiss_build_seconds += time.perf_counter() - start

            all_pids.extend(ids.tolist())
            progress.update(len(rows))
            batch_number += 1

    pids = np.asarray(all_pids, dtype=np.int64)
    assert index.ntotal == len(pids)

    faiss.write_index(index, str(OUTPUT / "index.faiss"))
    np.save(OUTPUT / "passage_ids.npy", pids)

    # Use CPU query embeddings, as planned for the frontend.
    # This leaves GPU memory available for Qwen later.
    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    queries = list(read_tsv(DATA / "questions.search.tsv"))

    # Warm up CPU inference before measuring query latency.
    model.encode(
        ["query: warmup"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    latencies = []

    with (OUTPUT / "rankings.tsv").open(
        "w", encoding="utf-8"
    ) as rankings:
        for qid, question in tqdm(queries, desc="Semantic retrieval"):
            start = time.perf_counter()

            query_vector = model.encode(
                ["query: " + question],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            query_vector = np.ascontiguousarray(
                query_vector, dtype=np.float32
            )

            scores, positions = index.search(query_vector, 100)

            # Order returned ties by original passage ID.
            matches = [
                (int(pids[position]), float(score))
                for position, score in zip(positions[0], scores[0])
                if position >= 0
            ]
            matches.sort(key=lambda item: (-item[1], item[0]))

            latencies.append(time.perf_counter() - start)

            for rank, (pid, score) in enumerate(matches, start=1):
                rankings.write(f"{qid}\t{pid}\t{rank}\t{score:.9g}\n")

    metrics = {
        "method": "semantic",
        "domain": "lifestyle",
        "split": "dev",
        "query_type": "search",
        "model": MODEL_NAME,
        "embedding_dimensions": DIMENSIONS,
        "max_input_tokens": MAX_LENGTH,
        "long_passage_handling": "truncate to model input limit",
        "passages": len(pids),
        "queries": len(queries),
        "candidate_limit": 100,
        "index_type": "FAISS IndexFlatIP",
        "normalized_embeddings": True,
        "passage_embedding_device": "cuda",
        "query_embedding_device": "cpu",
        "cpu_threads": 4,
        "embedding_seconds": embedding_seconds,
        "faiss_build_seconds": faiss_build_seconds,
        "indexing_seconds": embedding_seconds + faiss_build_seconds,
        "indexing_timing_scope": (
            "encoding and FAISS add; excludes download, loading and file I/O; "
            "reuses recorded encoding times for resumed batches"
        ),
        "mean_query_latency_ms": float(np.mean(latencies) * 1000),
    }

    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\nSemantic retrieval complete")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved files: {OUTPUT}")


if __name__ == "__main__":
    main()