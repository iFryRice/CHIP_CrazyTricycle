set -euo pipefail
cd /home/dcf/chip2026
export UV_CACHE_DIR=/home/dcf/chip2026/.uv-cache
export UV_PYTHON_INSTALL_DIR=/home/dcf/chip2026/.tools/python
export HF_HOME=/home/dcf/chip2026/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
.tools/uv run --no-sync python scripts/check_training_environment.py
