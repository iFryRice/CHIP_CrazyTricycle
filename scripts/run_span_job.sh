#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
name="${1:?A descriptive run name is required}"
shift
[[ "$name" =~ ^[a-z0-9_-]+$ ]] || { echo "Invalid run name" >&2; exit 2; }
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
gpu="${SPAN_GPU_INDEX:-2}"
[[ "$gpu" == "0" || "$gpu" == "2" ]] || { echo "Only declared GPU profiles 0 and 2 are supported" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p work/span_ner reports/span_ner
exec 9>"work/span_ner/gpu${gpu}.lock"
flock -n 9 || { echo "Another project job holds the GPU $gpu lock" >&2; exit 3; }
test ! -e "work/span_ner/$name"
free_kib=$(df --output=avail /home/dcf | tail -n 1 | tr -d ' ')
(( free_kib >= 20 * 1024 * 1024 )) || { echo "Less than 20 GiB free" >&2; exit 4; }
compute=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
[[ -z "$compute" ]] || { echo "GPU $gpu already has a compute process: $compute" >&2; exit 5; }
date -Is
.tools/uv run --no-sync python scripts/train_span_ner.py \
  --model-path models/biomedbert --threads 2 --max-span-width 64 \
  "$@" --output-dir "work/span_ner/$name" 2>&1 | tee "reports/span_ner/$name.log"
