#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for name in fold0 boundary-cap100-fold0 boundary-ranking-fold0; do
  .tools/uv run --no-sync python scripts/evaluate_linked_pipeline.py \
    --run-dir "work/span_ner/$name" --umls-terms resources/umls_hpo_terms.tsv.gz \
    --output-dir "reports/span_ner/linked_$name" \
    2>&1 | tee "reports/span_ner/linked_$name.log"
done
