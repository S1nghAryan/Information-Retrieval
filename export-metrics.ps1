$ErrorActionPreference = 'Stop'
$methods = @('tfidf','bm25','semantic_chunked','hybrid_chunked_two')
$data = @{}
foreach ($split in @('dev','test')) {
  $data[$split] = @()
  $splitMethods = if ($split -eq 'dev') { $methods + @('hybrid') } else { $methods }
  foreach ($method in $splitMethods) {
    $folder = Join-Path (Join-Path $PSScriptRoot 'artifacts\lifestyle') "$split\$method"
    $metrics = Get-Content -LiteralPath (Join-Path $folder 'metrics.json') -Raw | ConvertFrom-Json
    $curve = @(Import-Csv -LiteralPath (Join-Path $folder 'precision_recall.csv'))
    $data[$split] += @{ method=$method; metrics=$metrics; curve=$curve }
  }
}
$data | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'public\metrics.json') -Encoding utf8
