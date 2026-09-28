#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_DIR"
export UV_CACHE_DIR="$TASK_DIR/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$TASK_DIR/.tools/python"
export UV_CONCURRENT_DOWNLOADS=2
export UV_HTTP_TIMEOUT=120
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export HF_HOME="$TASK_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

test -x .tools/uv
test -f experiments/span_ner/environment.json
test -f uv.lock
available_kib="$(df -Pk "$TASK_DIR" | awk 'NR == 2 { print $4 }')"
if (( available_kib < 20 * 1024 * 1024 )); then
    echo 'Less than 20 GiB free disk; refusing environment setup.' >&2
    exit 1
fi

.tools/uv sync --locked --extra training --python 3.12
.tools/uv run --locked --extra training python -c 'import sys, torch, transformers; print("IMPORT_WITNESS", sys.version.split()[0], torch.__version__, transformers.__version__); print("COMPILED_CUDA_ARCHS", torch.cuda.get_arch_list())'
echo 'Environment installation completed; GPU and training smoke checks remain required.'
