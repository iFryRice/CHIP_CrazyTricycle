#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
gpu="${LLM_GPU_INDEX:-2}"
[[ "$gpu" == 0 || "$gpu" == 2 ]] || { echo 'Only GPU 0 and GPU 2 are declared'; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec 9>"work/span_ner/gpu${gpu}.lock"
flock -n 9 || { echo "A project job already holds GPU $gpu"; exit 3; }
compute=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
[[ -z "$compute" ]] || { echo "GPU $gpu already has a compute process"; exit 5; }
.venv/bin/python -u scripts/run_joint_review.py infer "$@"
