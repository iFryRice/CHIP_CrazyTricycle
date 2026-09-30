#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec 9>work/span_ner/gpu2.lock
flock -n 9 || { echo 'GPU 2 is held by another project job'; exit 3; }
compute=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader)
[[ -z "$compute" ]] || { echo 'GPU 2 already has a compute process'; exit 5; }
.venv/bin/python -c 'import json; p=json.load(open("reports/bio_corroboration/confirmation/summary.json")); assert p["promoted"] and all(p["checks"].values())'
.venv/bin/python -u scripts/train_bio_ner.py --fit-all --output-dir work/span_ner/bio-final-all80 2>&1 | tee reports/bio_ner/final-all80.log
.venv/bin/python -u scripts/predict_bio_supported_b.py 2>&1 | tee reports/bio_corroboration_b.log
