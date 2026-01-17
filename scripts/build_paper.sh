#!/usr/bin/env bash
set -euo pipefail

TEX_FILE="${1:-documents/main.tex}"
OUT_DIR="${2:-documents/build}"

mkdir -p "$OUT_DIR"

TEX_DIR="$(cd "$(dirname "$TEX_FILE")" && pwd)"
TEX_NAME="$(basename "$TEX_FILE")"
JOB_NAME="${TEX_NAME%.tex}"

pushd "$TEX_DIR" >/dev/null

pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$(cd "$(dirname "$OUT_DIR")" && pwd)/$(basename "$OUT_DIR")" "$TEX_NAME"

BUILD_DIR="$(cd "$(dirname "$OUT_DIR")" && pwd)/$(basename "$OUT_DIR")"
if [[ -f "$BUILD_DIR/$JOB_NAME.aux" ]]; then
  if grep -qE '\\\\citation|\\\\bibdata' "$BUILD_DIR/$JOB_NAME.aux"; then
    (cd "$BUILD_DIR" && bibtex "$JOB_NAME")
  fi
fi

pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" "$TEX_NAME"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" "$TEX_NAME"

echo "Built: $BUILD_DIR/$JOB_NAME.pdf"

popd >/dev/null
