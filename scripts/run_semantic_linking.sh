#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p work/span_ner/semantic_linking
exec 9>work/span_ner/gpu2.lock
flock -n 9 || { echo 'A project job already holds GPU 2'; exit 3; }
compute=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader)
[[ -z "$compute" ]] || { echo 'GPU 2 already has a compute process'; exit 5; }
.venv/bin/python -u scripts/evaluate_semantic_linking.py 2>&1 | tee work/span_ner/semantic_linking/run.log
