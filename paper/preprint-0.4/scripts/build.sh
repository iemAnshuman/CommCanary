#!/bin/sh
set -eu

SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SOURCE_DIR/../.." && pwd)
OUTPUT_DIR="$REPOSITORY_ROOT/output/pdf"
OUTPUT_PDF="$OUTPUT_DIR/commcanary-preprint-0.4-source-reconstruction.pdf"

mkdir -p "$OUTPUT_DIR"
python3 "$SOURCE_DIR/scripts/verify_sources.py"
python3 "$SOURCE_DIR/scripts/generate_figures.py"

(
  cd "$SOURCE_DIR"
  SOURCE_DATE_EPOCH=1785790552 tectonic main.tex \
    --outdir "$OUTPUT_DIR" --keep-logs
)

mv "$OUTPUT_DIR/main.pdf" "$OUTPUT_PDF"
printf '%s\n' "wrote $OUTPUT_PDF"
