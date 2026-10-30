#!/bin/sh
set -eu

SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SOURCE_DIR/../.." && pwd)
OUTPUT_DIR="$REPOSITORY_ROOT/output/pdf"
OUTPUT_PDF="$OUTPUT_DIR/commcanary-arxiv.pdf"

mkdir -p "$OUTPUT_DIR"
python3 "$SOURCE_DIR/scripts/verify_sources.py"
python3 "$SOURCE_DIR/scripts/trusted_join_uncertainty.py" --check
python3 "$SOURCE_DIR/scripts/decision_regret.py" --check
python3 "$SOURCE_DIR/scripts/generate_figures.py"

(
  cd "$SOURCE_DIR"
  SOURCE_DATE_EPOCH=1791072000 tectonic main.tex \
    --outdir "$OUTPUT_DIR" --keep-logs --keep-intermediates
)

if grep -E 'Overfull \\[hv]box|LaTeX Warning: (Citation|Reference).*undefined|There were undefined references' "$OUTPUT_DIR/main.log"; then
  printf '%s\n' 'paper build failed: unresolved references or overflowing content' >&2
  exit 1
fi

mv "$OUTPUT_DIR/main.pdf" "$OUTPUT_PDF"
printf '%s\n' "wrote $OUTPUT_PDF"
