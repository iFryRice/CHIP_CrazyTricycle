#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_DIR"
export UV_CACHE_DIR="$TASK_DIR/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$TASK_DIR/.tools/python"
export HF_HOME="$TASK_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

check_resources() {
    local available_kib active_compute
    available_kib="$(df -Pk "$TASK_DIR" | awk 'NR == 2 { print $4 }')"
    if (( available_kib < 20 * 1024 * 1024 )); then
        echo 'Less than 20 GiB free disk; training was not started.' >&2
        return 1
    fi
    active_compute="$(nvidia-smi --id=2 --query-compute-apps=pid --format=csv,noheader,nounits)"
    if [[ "$active_compute" =~ [0-9] ]]; then
        echo 'GPU 2 has an existing compute process; training was not started.' >&2
        return 1
    fi
}

check_resources
.tools/uv run --no-sync python scripts/train_span_ner.py \
    --model-path models/biomedbert --output-dir work/span_ner/smoke_fold0 \
    --fold 0 --smoke --threads 2 --max-span-width 64 \
    2>&1 | tee reports/span_ner/smoke_fold0.log

.venv/bin/python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path('work/span_ner/smoke_fold0/summary.json').read_text())
if not (summary['status'] == 'completed' and summary['smoke']
        and summary['optimizer_steps'] == 20):
    raise RuntimeError('The 20-step training smoke did not complete successfully')
print('SMOKE_ACCEPTED', summary['duration_seconds'])
PY

check_resources
.tools/uv run --no-sync python scripts/train_span_ner.py \
    --model-path models/biomedbert --output-dir work/span_ner/fold0 \
    --fold 0 --epochs 5 --threads 2 --max-span-width 64 \
    2>&1 | tee reports/span_ner/fold0.log
