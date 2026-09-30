#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -u scripts/confirm_bio_corroboration.py 2>&1 | tee reports/bio_confirmation.log
