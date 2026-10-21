$ErrorActionPreference = 'Stop'
$ragProjectRoot = if ($env:RAG_ROOT) { $env:RAG_ROOT } else { $PSScriptRoot }
$ragPython = Join-Path $ragProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ragPython)) { throw "Existing RAG Python environment not found: $ragPython. Set RAG_ROOT to your project folder." }
& $ragPython (Join-Path $PSScriptRoot 'backend\server.py')
