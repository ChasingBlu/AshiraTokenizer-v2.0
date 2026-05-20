param(
    [string]$Corpus = "D:\ChasingBlu_RND\Lab\Active\ODE_transformer\tmp\tok_smoke_corpus",
    [string]$Output = ".\runs\smoke_out"
)

$ErrorActionPreference = "Stop"
$Exe = ".\target\release\ashira_tokenizer_v2.exe"

if (!(Test-Path $Exe)) {
    throw "Trainer binary not found: $Exe. Run orchestration/build_release.ps1 first."
}

New-Item -ItemType Directory -Force -Path $Output | Out-Null

& $Exe `
  --corpus $Corpus `
  --output $Output `
  --vocab-size 320 `
  --min-freq 2 `
  --accelerator cpu

if ($LASTEXITCODE -ne 0) {
    throw "Smoke run failed"
}

Write-Host "[SMOKE] PASS"

