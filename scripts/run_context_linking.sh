#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p reports/association_context
.venv/bin/python -u scripts/evaluate_context_linking.py 2>&1 | tee reports/association_context/run.log
