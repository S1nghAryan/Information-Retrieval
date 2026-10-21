param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$ragApp = $PSScriptRoot
$ragLogs = Join-Path $ragApp 'logs'

function Get-RagJson([string]$Url) {
    try { return Invoke-RestMethod -Uri $Url -TimeoutSec 3 } catch { return $null }
}
function Wait-RagService([string]$Url, [string]$Name, [int]$Seconds = 60) {
    $ragDeadline = (Get-Date).AddSeconds($Seconds)
    do {
        $ragResponse = Get-RagJson $Url
        if ($null -ne $ragResponse) { return $ragResponse }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $ragDeadline)
    throw "$Name did not become ready. Check $ragLogs for details."
}
function Assert-RagPortFree([int]$Port) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port is occupied by an unrecognized or unhealthy service. Close that service and retry."
    }
}
function Start-RagService([string]$Executable, [string]$Arguments, [string]$Name) {
    Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $ragApp -WindowStyle Hidden -RedirectStandardOutput (Join-Path $ragLogs "$Name.log") -RedirectStandardError (Join-Path $ragLogs "$Name-error.log") | Out-Null
}

try {
    New-Item -ItemType Directory -Path $ragLogs -Force | Out-Null
    Write-Host 'Starting Fieldnotes...'
    $ragRoot = if ($env:RAG_ROOT) { $env:RAG_ROOT } else { $PSScriptRoot }
    $ragPython = Join-Path $ragRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $ragPython)) { throw "Python environment missing at $ragPython. Set RAG_ROOT to your original RAG project folder. See README.md for setup." }
    $ragNode = (Get-Command node.exe -ErrorAction Stop).Source
    $ragVite = Join-Path $ragApp 'node_modules\vite\bin\vite.js'
    if (-not (Test-Path -LiteralPath $ragVite)) { throw 'Frontend dependencies are missing. Run npm ci in this folder once, then launch again.' }

    $ragModels = Get-RagJson 'http://127.0.0.1:11434/api/tags'
    if ($null -eq $ragModels) {
        Assert-RagPortFree 11434
        $ragOllama = (Get-Command ollama.exe -ErrorAction Stop).Source
        Write-Host 'Starting Ollama...'
        Start-RagService $ragOllama 'serve' 'ollama'
        $ragModels = Wait-RagService 'http://127.0.0.1:11434/api/tags' 'Ollama'
    }
    if (-not ($ragModels.models | Where-Object { $_.name -eq 'qwen3:4b' })) { throw 'The qwen3:4b model is missing. Run ollama pull qwen3:4b once, then launch again.' }

    $ragHealth = Get-RagJson 'http://127.0.0.1:8000/api/health'
    if ($null -eq $ragHealth -or $ragHealth.status -ne 'ok' -or $ragHealth.model -ne 'qwen3:4b') {
        Assert-RagPortFree 8000
        Write-Host 'Starting the retrieval backend...'
        Start-RagService $ragPython ('"' + (Join-Path $ragApp 'backend\server.py') + '"') 'backend'
        $ragHealth = Wait-RagService 'http://127.0.0.1:8000/api/health' 'Backend' 90
    }

    $ragFrontend = Get-RagJson 'http://127.0.0.1:5173/'
    if ($null -eq $ragFrontend -or [string]$ragFrontend -notmatch 'Fieldnotes') {
        Assert-RagPortFree 5173
        Write-Host 'Starting the website...'
        Start-RagService $ragNode ('"' + $ragVite + '" --host 127.0.0.1 --port 5173 --strictPort') 'frontend'
        $null = Wait-RagService 'http://127.0.0.1:5173/' 'Website'
    }
    $ragProxy = Wait-RagService 'http://127.0.0.1:5173/api/health' 'Website backend connection'
    if ($ragProxy.status -ne 'ok' -or -not $ragProxy.ollama_available) { throw 'The website is running, but the backend or local model is not ready. Check logs and retry.' }
    Write-Host 'Fieldnotes is ready: http://127.0.0.1:5173/'
    Write-Host 'Existing services are reused. Background services stay running after this window closes.'
    if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:5173/' }
    exit 0
} catch {
    Write-Host ("Startup failed: " + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
