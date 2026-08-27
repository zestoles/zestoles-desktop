# Starts the optional loopback-only SearXNG service with a per-install secret.
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$environmentFile = Join-Path $root '.env'

if (-not (Get-Command docker.exe -ErrorAction SilentlyContinue)) {
    throw 'Docker Desktop is required for the optional SearXNG service.'
}

if (-not (Test-Path -LiteralPath $environmentFile)) {
    $bytes = New-Object byte[] 32
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $secret = -join ($bytes | ForEach-Object { $_.ToString('x2') })
    [IO.File]::WriteAllText($environmentFile, "SEARXNG_SECRET=$secret`r`n",
        [Text.UTF8Encoding]::new($false))
}

Push-Location $root
try {
    & docker.exe compose up -d
    if ($LASTEXITCODE -ne 0) { throw "docker compose failed ($LASTEXITCODE)" }
}
finally {
    Pop-Location
}

Write-Host 'SearXNG is available at http://127.0.0.1:8888/' -ForegroundColor Green
