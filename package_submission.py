"""Package source and evidence without datasets, indexes, models or dependencies."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
OUT=HERE/'submission'
ROOT=HERE
reference=OUT/'experiment-source'
reference.mkdir(exist_ok=True)
for source in [*ROOT.glob('*.py'), ROOT/'processss.txt']:
    shutil.copy2(source,reference/source.name)
data=json.loads((HERE/'public/metrics.json').read_text(encoding='utf-8-sig'))
for split,rows in data.items():
    for row in rows:
        folder=ROOT/'artifacts/lifestyle'/split/row['method']
        dest=OUT/'evaluation-evidence'/split/row['method']
        dest.mkdir(parents=True,exist_ok=True)
        for name in ['metrics.json','precision_recall.csv','per_query_metrics.csv']:
            shutil.copy2(folder/name,dest/name)
deps=sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions(path=[str(HERE/'backend/vendor')]))
(OUT/'backend-dependencies.lock.txt').write_text('\n'.join(deps)+'\n',encoding='utf-8')
files=[]
for folder in ['src','public','backend','submission']:
    for p in (HERE/folder).rglob('*'):
        if p.is_file() and not any(x in p.parts for x in ['vendor','models','__pycache__']) and p.suffix not in ['.zip','.log'] and p.name!='package-manifest.json':
            files.append(p)
for name in ['package.json','package-lock.json','index.html','vite.config.js','README.md','CONTEXT.md','VERIFICATION.md','DEMO-CHECKLIST.md','REMAINING-STEPS.md','start-backend.ps1','Run-Fieldnotes.bat','Run-Fieldnotes.ps1','requirements.txt','export-metrics.ps1','build_report.py','package_submission.py','.gitignore']:
    files.append(HERE/name)
manifest=[{'path':str(p.relative_to(HERE)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
manifest_path=OUT/'package-manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
files.append(manifest_path)
archive=OUT/'fieldnotes-rag-submission.zip'
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
    for p in files:z.write(p,str(Path('fieldnotes-rag')/p.relative_to(HERE)))
with ZipFile(archive) as z:
    assert z.testzip() is None
    for row in manifest:
        assert hashlib.sha256(z.read('fieldnotes-rag/'+row['path'])).hexdigest()==row['sha256']
print(f'Verified {len(files)} packaged files; {archive.stat().st_size:,} bytes.')
