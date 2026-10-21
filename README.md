# Fieldnotes — Lifestyle RAG

React/Vite frontend and local FastAPI backend, with the existing light cartoon-inspired theme.

## Five methods

- TF-IDF.
- BM25 (Lucene, k1=1.5, b=0.75).
- Sentence Transformer (E5, overlapping document chunks).
- RRF (BM25 + semantic), the default: weights 1:4, k=60.
- Standard RRF (all three): TF-IDF + BM25 + original Sentence Transformer, equal weights, k=60.

The semantic and RRF versions were retained for stronger existing nDCG@10. This is not a claim that they win every metric. The default RRF formula is a weighted two-component variation of the PPT's standard three-component formula. TF-IDF remains available independently. Standard three-way RRF is also available to demonstrate the PPT formula. Other duplicate versions and the cross-encoder remain removed.

## Run

Double-click `Run-Fieldnotes.bat` in this folder to start Ollama, the backend, and the frontend, then open the website. Existing healthy services are reused; background services remain running after the launcher closes. Startup logs are stored in `logs/`. Keep `Run-Fieldnotes.ps1` beside the batch file. The launcher uses the installed dependencies, original RAG environment, and downloaded qwen3:4b model; it reports missing setup instead of downloading automatically.

Manual startup (alternative):

From this directory, run `npm ci` then `npm run dev`.
In a second terminal, run `./start-backend.ps1`. Ollama must have `qwen3:4b` available.

All application and retrieval code now lives in this repository. Paths default to this folder. The backend also accepts RAG_ROOT for a separately located data/index folder.

If `backend/vendor` is missing:

```powershell
& .\.venv\Scripts\python.exe -m pip install --target backend/vendor -r backend/requirements.txt
```

The backend listens at 127.0.0.1:8000; Vite forwards `/api`. The app searches Lifestyle dev. The dashboard has five development methods and four test methods; standard RRF has no saved test evaluation. `export-metrics.ps1` refreshes these measured rows from original artifacts. Original indexes, experiments and synced project references remain read-only.

`npm run build` produces `dist`. A deployed frontend requires an API proxy to FastAPI; a standalone static host is insufficient.

## Answers and evaluation

Retrieval retains 100 candidates, displays ten sources and supplies excerpts from all ten displayed results to Qwen. Each excerpt is bounded to 550 UTF-8 bytes. Qwen considers all supplied excerpts and selects up to five complementary quotations within an 8192-token context window; the server validates exact matches and source IDs, retries invalid output once by asking Qwen to select numbered complete source sentences, then copying their exact text, and reports failure or abstention explicitly. This is extractive LLM use, not free-form synthesis. Provenance does not establish source truth, relevance or completeness.

Index loading, search and generation timings are separated. Search excludes source-text lookup; saved complete RRF latency remains unmeasured. Cold starts are much slower than warm queries.

Run `backend/check_integration.py` against the running services for five-method parity and removed-method rejection checks; `backend/test_generation.py` for deterministic quotation validation; and `backend/validate_demo.py` for bounded live answer regressions.

`backend/evaluate_query_dev.py` tested query-cleaning variants within TF-IDF, BM25 and RRF. The unchanged policy won each selection comparison; no new setting was promoted. Results are in `submission/query-experiment-dev.md`. Development data had been used previously; these are exploratory results. Do not tune using already-inspected test scores.

See `REMAINING-STEPS.md`, `DEMO-CHECKLIST.md`, and `submission/technical-report.html`. The submission excludes dependencies, models, datasets and historical evidence for removed methods.

Retrieved sources highlight verified answer quotations in pink and exact matching question/answer keywords in lavender. Keyword matches are visual aids, not relevance judgments.

## Setup after cloning from GitHub (Windows)

Install Python 3.12, Node.js 22.12+ and Ollama, then in the repository folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm ci
ollama pull qwen3:4b
```

Restore `data/` and `artifacts/` from your separately shared project assets, preserving their relative paths. The app needs the Lifestyle corpus, lexical indexes, semantic indexes, passage mappings and locally saved E5 embedding model. They are excluded from Git because of their size. A source-only clone cannot answer queries until these assets are restored. Existing baseline and evaluation scripts are included for rebuilding experiments; saved metrics in `public/` support the dashboard.

Then run `.\Run-Fieldnotes.bat`. No personal absolute path is required. `requirements.txt` lists the runtime packages; installing on a fresh machine has not been verified. Historical submission documents may describe earlier app behavior; use this README and live code for current setup.

Before uploading, review `git status` and keep `.venv/`, `node_modules/`, `data/`, `artifacts/`, logs and secrets excluded. No GitHub upload is performed by the launcher.
