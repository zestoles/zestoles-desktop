# Installs ZESTOLES local runtimes and creates its desktop shortcut.
# ASCII-only for Windows PowerShell 5.1 on Turkish Windows installations.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$commit = '5de7a54aa4e5e2baadb0182dde554908b48b85c2'

function Step($text) { Write-Host "`n== $text ==" -ForegroundColor Cyan }

Step 'On kosullar denetleniyor'
$uv = Get-Command uv.exe -ErrorAction SilentlyContinue
if (-not $uv) {
    throw 'uv bulunamadi. Once su komutu calistirin: winget install astral-sh.uv'
}
$ollama = Get-Command ollama.exe -ErrorAction SilentlyContinue
if (-not $ollama) {
    throw 'Ollama bulunamadi. Once https://ollama.com/download/windows adresinden kurun.'
}

Step 'Ana Python 3.12 ortami kuruluyor'
if (-not (Test-Path '.venv\Scripts\python.exe')) { & $uv.Source venv --python 3.12 .venv }
& $uv.Source pip install --python .venv\Scripts\python.exe -r requirements-core.txt
& $uv.Source pip install --python .venv\Scripts\python.exe -r requirements-voice.txt

Step 'Turkce konusma tanima modeli indiriliyor'
& .\.venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', download_root='data/ses/whisper')"

Step 'Yerel referans sesi hazirlaniyor'
$reference = Join-Path $root 'data\ses\referans\varsayilan.wav'
if (-not (Test-Path $reference)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $reference) | Out-Null
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try {
        $turkish = $synth.GetInstalledVoices() |
            Where-Object { $_.VoiceInfo.Culture.Name -eq 'tr-TR' } |
            Select-Object -First 1
        if ($turkish) { $synth.SelectVoice($turkish.VoiceInfo.Name) }
        $synth.Rate = -1
        $synth.SetOutputToWaveFile($reference)
        $synth.Speak('Merhaba. Ben Zestoles. Hazirim ve seni dikkatle dinliyorum.')
    }
    finally {
        $synth.SetOutputToNull()
        $synth.Dispose()
    }
}

Step 'Dogal Chatterbox V3 ses ortami kuruluyor'
$ttsPython = 'data\ses\chatterbox-v3-env\Scripts\python.exe'
if (-not (Test-Path $ttsPython)) { & $uv.Source venv --python 3.12 data\ses\chatterbox-v3-env }
& $uv.Source pip install --python $ttsPython "git+https://github.com/resemble-ai/chatterbox.git@$commit"
& $uv.Source pip install --python $ttsPython --upgrade torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

Step 'Yerel modeller indiriliyor'
& $ollama.Source pull qwen3.5:9b
& $ollama.Source pull qwen3:14b
& $ollama.Source pull bge-m3

Step 'Kurulum dogrulaniyor'
& .\.venv\Scripts\python.exe -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { throw "Testler basarisiz (kod $LASTEXITCODE)" }

Step 'Masaustu kisayolu olusturuluyor'
& (Join-Path $PSScriptRoot 'install-zestoles-shortcut.ps1')

Write-Host "`nZESTOLES hazir. Masaustundeki ZESTOLES kisayolunu acin." -ForegroundColor Green
