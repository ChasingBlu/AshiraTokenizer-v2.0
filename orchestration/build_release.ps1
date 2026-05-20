$ErrorActionPreference = "Stop"

$Cargo = "C:\Users\Op-Prime\.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin\cargo.exe"
if (!(Test-Path $Cargo)) {
    throw "Cargo not found at $Cargo"
}

Write-Host "[BUILD] Using cargo at $Cargo"
& $Cargo build --release
if ($LASTEXITCODE -ne 0) {
    throw "Release build failed"
}

Write-Host "[BUILD] PASS"

