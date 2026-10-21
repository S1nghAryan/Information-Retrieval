"""Build a self-contained technical report from saved metrics, without rerunning experiments."""
from pathlib import Path
import base64
import csv
import hashlib
import html
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'submission'
OUT.mkdir(exist_ok=True)
DATA = json.loads((HERE / 'public/metrics.json').read_text(encoding='utf-8-sig'))
NAMES = {'tfidf':'TF-IDF','bm25':'BM25','semantic_chunked':'Sentence Transformer','hybrid_chunked_two':'RRF (BM25 + semantic)','hybrid':'Standard RRF (all three)'}
COLORS = {'tfidf':'#34374c','bm25':'#777ca3','semantic_chunked':'#2c2e3e','hybrid_chunked_two':'#ee2b47','hybrid':'#a16480'}
KEYS = ['precision_at_10','success_at_5','recall_at_10','recall_at_100','map_at_100','ndcg_at_10']
HEADINGS = ['Method','P@10','S@5','R@10','R@100','MAP@100','nDCG@10']
figures = {}
tables = {}
for split, rows in DATA.items():
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for i, row in enumerate(rows):
        x = [float(p['recall']) for p in row['curve']]
        y = [float(p['interpolated_precision']) for p in row['curve']]
        assert len(x) == 101 and all(0 <= v <= 1 for v in y)
        ax.plot(x,y,label=NAMES[row['method']],color=COLORS[row['method']],linestyle=['-','--',':','-.'][i%4],linewidth=1.9)
    ax.set(xlabel='Recall',ylabel='Interpolated precision',xlim=(0,1),ylim=(0,1),title=f'LoTTE Lifestyle {split} search queries')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9,loc='upper right',framealpha=.95)
    fig.tight_layout()
    file=OUT / f'precision-recall-{split}.png'
    fig.savefig(file,dpi=170)
    plt.close(fig)
    figures[split]='data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode()
    table_rows=[[NAMES[r['method']]]+[f"{r['metrics'][k]:.6f}" for k in KEYS] for r in rows]
    tables[split]='<table><thead><tr>'+''.join(f'<th>{h}</th>' for h in HEADINGS)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in table_rows)+'</tbody></table>'
    with (OUT/f'{split}-comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['method']+KEYS)
        for r in rows:writer.writerow([r['method']]+[r['metrics'][k] for k in KEYS])

answer=json.loads((HERE/'backend/smoke-answer.json').read_text(encoding='utf-8-sig'))
example=html.escape(answer['answer'])
sources=''.join(f"<li><strong>[{s['rank']}] Passage {s['passage_id']}</strong><p>{html.escape(s['text'])}</p></li>" for s in answer['context_sources'])
timing='<table><thead><tr><th>Method</th><th>Dev mean ms</th><th>Test mean ms</th></tr></thead><tbody>'
for method in NAMES:
    cells=[]
    for split in ['dev','test']:
        row=next((r for r in DATA[split] if r['method']==method),None)
        v=row['metrics'].get('mean_query_latency_ms') if row else None
        cells.append(f'{v:.2f}' if isinstance(v,(int,float)) else 'Not measured' if row else 'Not evaluated')
    timing+=f'<tr><td>{NAMES[method]}</td><td>{cells[0]}</td><td>{cells[1]}</td></tr>'
timing+='</tbody></table>'
body=f'''
<h1>Hybrid retrieval and source-backed answers on LoTTE Lifestyle</h1>
<p class="subtitle">Technical report · 20 October 2026<br>Aryan Singh · Aryan Masrani · Chirag Jotwani</p>
<h2>Abstract</h2>
<p>The application exposes five retrieval choices: TF-IDF, BM25, Sentence Transformer, weighted two-way RRF and standard three-way RRF. The retained semantic method uses overlapping document chunks. RRF combines BM25 and semantic rankings with weights 1:4 and k=60. This configuration was retained by the user's choice of the stronger existing nDCG@10 version. Its saved test nDCG@10 is 0.615327, compared with 0.602541 for semantic retrieval. These descriptive results do not establish statistical significance or generated-answer accuracy.</p>
<h2>1 Assignment and scope</h2>
<p>The supplied assignment requests document chunking, tokenization, stemming, sparse and vector indexing, TF-IDF, BM25, Sentence Transformer, RRF context selection, LLM generation, a working search interface, performance tables and precision–recall graphs. The current scope is the Lifestyle domain, search queries only. The live app searches dev data. Standard RRF follows the PPT formula with equal contributions from TF-IDF, BM25 and original Sentence Transformer rankings. The existing weighted two-way RRF remains available and is the current default.</p>
<table><thead><tr><th>Split</th><th>Passages</th><th>Search queries</th></tr></thead><tbody><tr><td>Development</td><td>268,893</td><td>417</td></tr><tr><td>Test</td><td>119,461</td><td>661</td></tr></tbody></table>
<h2>2 Preprocessing and retrieval</h2>
<p>Lexical processing lowercases, tokenizes words and applies English stemming. Stopwords are retained. TF-IDF uses L2 normalization, sublinear term frequency and smoothed IDF, with a transposed CSR sparse postings matrix. BM25 uses the saved Lucene variant, k1=1.5 and b=0.75. Positive matches are sorted by score, with ascending passage ID for ties. Ranking does not consult relevance labels.</p>
<p>Sentence Transformer uses the saved intfloat/e5-small-v2 model, query/passage prefixes (query: and passage:), normalized 384-dimensional embeddings and exact FAISS inner-product search. Long documents use windows of up to 480 body tokens with 64-token overlap, checked against the 512-token input limit. Each original passage receives its maximum chunk similarity. Adaptive expansion returns 100 unique passage IDs. The dev index contains 295,689 vectors; embeddings of short passages were reused. No additional reranking model is used.</p>
<h2>3 Standard and weighted RRF</h2>
<p>For document d: score(d) = 1/(60 + BM25 rank(d)) + 4/(60 + semantic rank(d)). A missing component contributes zero. Fusion consumes the two top-100 lists and retains 100 results, breaking ties by passage ID. This version has higher nDCG@10 than the existing three-way weighted alternative on both saved splits; it is not best on every metric. The method and reported original scores were frozen before the original test evaluation. Test scores have since been inspected and cannot be reused to tune new settings.</p>
<p>Standard RRF sums 1/(60+rank) across the TF-IDF, BM25 and original semantic top-100 lists, without raw-score normalization. Missing components contribute zero; ties use passage ID. It uses the saved original semantic index and original passage context, while standalone Sentence Transformer and weighted RRF use chunks. Standard RRF has saved dev results only (nDCG@10 0.393792); no test score is inferred.</p>
<h2>4 LLM and context selection</h2>
<p>React/Vite calls FastAPI, which reads saved indexes and sends excerpts from the first three results to local Ollama qwen3:4b. Semantic matches use their best chunk when available. Each excerpt is limited to 550 UTF-8 bytes with sentence-aware clipping. Qwen produces structured evidence selections using temperature 0, a 4,096-token context and at most 768 output tokens. The server checks source IDs and exact quotation matches and expands partial selections to sentence boundaries. Invalid selections retry once; failures are explicit, and no-evidence selections abstain.</p>
<p>The LLM is active, but this is extractive selection rather than free-form synthesis. It is a narrower interpretation of the assignment's generation requirement. Exact quotations establish provenance, not source truth, relevance or completeness. Earlier prompting did not reliably prevent unsupported qualifiers; independent answer-quality evaluation is still needed before restoring synthesized prose.</p>
<h2>5 Evaluation definitions</h2>
<p>Binary relevance is macro-averaged over queries. P@10 divides relevant hits by ten. Recall@10 and Recall@100 divide by all labeled relevant passages. AP divides summed precision at relevant ranks by all labeled relevant passages; MAP averages AP. nDCG@10 uses binary gains and logarithmic discount with ideal ranking normalization. Success@5 is an additional diagnostic. Curves are 101-point interpolated precision–recall, truncated at 100 retrieved results. Missing positions are nonrelevant.</p>
<h2>6 Development results</h2>
{tables['dev']}
<figure><img src="{figures['dev']}" alt="Five-method development precision recall comparison"><figcaption>417 development queries. Saved original configurations.</figcaption></figure>
<h2>7 Test results</h2>
{tables['test']}
<figure><img src="{figures['test']}" alt="Four-method test precision recall comparison"><figcaption>661 test queries. Saved original configurations.</figcaption></figure>
<p>RRF's nDCG@10 advantage over semantic retrieval is 0.012786. Dev and test contain different collections and questions, so cross-split differences are not before/after improvements. The current interface offers five methods; dev tables show five and test tables show the four measured methods. Other historical results remain excluded.</p>
<h2>8 Attempts to improve existing methods</h2>
<p>A new development-only experiment compared unchanged query processing against removal of common question words and English stopwords (keeping not/no/never). It reused the fixed passage indexes, stemming and ranking formulas. TF-IDF and BM25 were evaluated directly; BM25 variants were also fused with the unchanged semantic ranking. Candidate selection used 333 dev questions; the remaining 84 were used to report the selected candidate. For all three comparisons, the unchanged policy won on selection nDCG@10. No new policy was deployed and no benchmark scores were replaced. This previously inspected development split is exploratory, not an independent holdout. The report and exact results are in query-experiment-dev.md/json.</p>
<p>The retained semantic chunking is already an improvement attempt within the requested Sentence Transformer method. New chunk sizes or embedding changes would require separate indexing and independent validation; they were not claimed as improvements here. No test data was read for the new query experiment.</p>
<h2>9 Timing</h2>
{timing}
<p>Saved hybrid timing does not measure complete retrieval and is shown as unmeasured. Live loading, search and generation are separated. Search excludes source lookup, transport and rendering. Cold index loading can be substantial; one observed keyboard test took 54.7 seconds to load indexes. These operational observations are not controlled latency benchmarks.</p>
<h2>10 Live example and checks</h2>
<p>A live standard-RRF question, “Which plants grow well indoors?”, returned the following Qwen-selected evidence:</p>
<blockquote>{example}</blockquote><ol class="sources">{sources}</ol>
<p>Five-method API parity, rejection of removed method identifiers and a live answer are checked by backend/check_integration.py. This is bounded integration coverage. The earlier main-flow keyboard checks cover submission, citations, disclosures, chart controls and visible focus; broader device/screen-reader coverage is not claimed. The reduced interface and production build are checked separately. See VERIFICATION.md for completed checks. Report print preview remains pending.</p>
<h2>11 Run and demonstrate</h2>
<ol><li>Keep the project-local Python environment, data, saved E5 model and indexes available. Set RAG_ROOT before start-backend.ps1 if relocating it.</li><li>Install frontend dependencies with npm ci; install backend/requirements.txt into backend/vendor using the original Python environment if needed.</li><li>Start Ollama with qwen3:4b, run start-backend.ps1, then npm run dev.</li><li>Demonstrate TF-IDF, BM25, Sentence Transformer, standard three-way RRF and weighted two-way RRF on a common question. Explain both fusion formulas.</li><li>Show source quotations, exact excerpts, performance tables, PR curves and CSV export. Add institution-required student identifiers and review the report print layout before submission.</li></ol>
<h2>References</h2>
<ul><li>Information Retrieval Project Assignments, supplied two-slide deck.</li><li><a href="https://huggingface.co/intfloat/e5-small-v2">E5 small v2 model card</a>.</li><li>Cormack, Clarke and Buettcher, SIGIR 2009: <a href="https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/">Reciprocal rank fusion</a>.</li><li><a href="https://github.com/stanford-futuredata/ColBERT">LoTTE benchmark resources</a>.</li><li>Local frozen metrics and evaluation CSVs; see source-manifest.json and evaluation-evidence.</li></ul>
'''

css='''body{font-family:Georgia,serif;color:#20222b;background:#e9eaf0;margin:0;font-size:16px;line-height:1.65}main{max-width:1050px;margin:35px auto;background:white;padding:55px 65px}h1{font-size:32px;line-height:1.2;color:black;margin:0 0 10px}h2{font-family:Arial,sans-serif;font-size:22px;color:black;margin-top:34px;line-height:1.3}.subtitle,figcaption{color:#555;font-size:14px}table{width:100%;border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px;margin:20px 0}th,td{border:1px solid #d9d9d9;padding:10px 7px;text-align:right}th{background:#34374c;color:white}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f0f1f5}figure{margin:25px 0}img{max-width:100%;height:auto}figcaption{line-height:1.45;margin-top:8px}blockquote{margin:18px 25px;font-style:italic}.sources{font-size:14px}.sources li{margin-bottom:15px}a{color:#34374c;overflow-wrap:anywhere}li{margin-bottom:9px}@media(max-width:700px){main{padding:25px 18px}table{font-size:10px}td,th{padding:7px 3px}}@media print{body{background:white;font-size:11pt}main{margin:0;padding:0;max-width:none}h2{break-after:avoid}figure,tr{break-inside:avoid}thead{display:table-header-group}a{color:black}table{font-size:9pt}@page{size:letter;margin:.7in}}'''
(OUT/'technical-report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LoTTE Lifestyle technical report</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>',encoding='utf-8')
manifest=[]
root=HERE
inputs=list(root.glob('*.py'))+[root/'processss.txt']
for split,rows in DATA.items():
    for row in rows:
        for name in ['metrics.json','precision_recall.csv']:
            inputs.append(root/'artifacts/lifestyle'/split/row['method']/name)
for p in inputs:
    manifest.append({'path':str(p.relative_to(root)) if p.is_relative_to(root) else str(p.relative_to(HERE)), 'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/'source-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Report, two PR figures, comparison CSVs and source manifest created.')
