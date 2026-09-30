#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p work/span_ner/umls_augmentation
.venv/bin/python -u scripts/optimize_umls_augmentation.py 2>&1 | tee work/span_ner/umls_augmentation/run.log
