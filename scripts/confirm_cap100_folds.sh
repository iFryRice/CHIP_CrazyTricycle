#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for fold in "$@"; do
  [[ "$fold" =~ ^[1-4]$ ]] || { echo "Expected a confirmation fold from 1 to 4" >&2; exit 2; }
  name="boundary-cap100-fold${fold}"
  bash scripts/run_span_job.sh "$name" --fold "$fold" --epochs 5 --pos-weight-cap 100 --no-eval-every-epoch
  .tools/uv run --no-sync python scripts/evaluate_linked_pipeline.py \
    --run-dir "work/span_ner/$name" --umls-terms resources/umls_hpo_terms.tsv.gz \
    --frozen-selection experiments/span_ner/selected_pipeline.json \
    --output-dir "reports/span_ner/confirmed_$name" \
    2>&1 | tee "reports/span_ner/confirmed_$name.log"
done
