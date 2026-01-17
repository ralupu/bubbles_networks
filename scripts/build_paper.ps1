param(
  [string]$TexFile = "documents/main.tex",
  [string]$OutDir = "documents/build"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $TexFile)) {
  throw "Missing TeX file: $TexFile"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$outDirPath = (Resolve-Path $OutDir).Path

$texPath = Resolve-Path $TexFile
$texDir = Split-Path $texPath -Parent
$texName = [System.IO.Path]::GetFileName($texPath)
$jobName = [System.IO.Path]::GetFileNameWithoutExtension($texPath)

Push-Location $texDir

try {
  Write-Host "Building PDF with pdflatex/bibtex into $outDirPath"

  & pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$outDirPath" "$texName" | Out-Host

  $auxPath = Join-Path $outDirPath "$jobName.aux"
  if (Test-Path $auxPath) {
    $auxText = Get-Content $auxPath -Raw
    if ($auxText -match "\\\\citation" -or $auxText -match "\\\\bibdata") {
      Write-Host "Running bibtex"
      Push-Location $outDirPath
      try {
        & bibtex "$jobName" | Out-Host
      } finally {
        Pop-Location
      }
    }
  }

  & pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$outDirPath" "$texName" | Out-Host
  & pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$outDirPath" "$texName" | Out-Host

  $pdf = Join-Path $outDirPath "$jobName.pdf"
  if (-not (Test-Path $pdf)) {
    throw "Expected PDF not found: $pdf"
  }
  Write-Host "Built: $pdf"
} finally {
  Pop-Location
}
