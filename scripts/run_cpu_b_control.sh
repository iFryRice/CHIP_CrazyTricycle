#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec .venv/bin/python -u -m patientphex run --target-set B \
  --target-file PatientPheX-V1-B/PatientPheX-V1-B.jsonl \
  --data-dir PatientPheX-V1-A --report-dir reports/cpu_b_control \
  --output submissions/patientphex_b_cpu.jsonl --folds 5 --seed 20260927
