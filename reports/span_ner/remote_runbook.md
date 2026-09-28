# Remote span training runbook

Host: `dcf@10.253.27.177`. All files physically reside under `/home/dcf`.
Project: `/home/dcf/chip2026`. Environment specification:
`experiments/span_ner/environment.json`. Use the project `.tools/uv` and `.venv`.
Do not use the system Python environment or modify other jobs.

## Acceptance commands

Run only after `reports/span_ner/environment_setup.log` reports installation
completion. Verify GPU 2 still has no compute process and sufficient free memory
before the CUDA witness. These commands must fail if the expected CUDA stack or
official model files are unavailable.

```bash
cd /home/dcf/chip2026
export UV_CACHE_DIR=/home/dcf/chip2026/.uv-cache
export UV_PYTHON_INSTALL_DIR=/home/dcf/chip2026/.tools/python
export HF_HOME=/home/dcf/chip2026/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
.tools/uv run --no-sync python scripts/verify_local_model.py --model-dir models/biomedbert --manifest reports/span_ner/model_source.json --report reports/span_ner/model_verification.json
.tools/uv run --no-sync python -m unittest discover -s tests -v
.tools/uv run --no-sync python scripts/check_training_environment.py
```

A fresh reviewer should run these exact commands, report every divergence, and
require all tests (including tensor tests) to run without skips. Import success
alone is insufficient. The expected kernel sentinel is `CUDA_WITNESS`.

## Short training and initial experiment

After acceptance passes, run the following with the same environment variables:

```bash
.tools/uv run --no-sync python scripts/train_span_ner.py --model-path models/biomedbert --output-dir work/span_ner/smoke_fold0 --fold 0 --smoke --threads 2 --max-span-width 64
```

The smoke run performs 20 optimizer steps and evaluates all 16 held-out documents.
Inspect `summary.json`, finite losses, effective updates, GPU memory and throughput.
Only after a successful smoke, start a separate five-epoch run:

```bash
.tools/uv run --no-sync python scripts/train_span_ner.py --model-path models/biomedbert --output-dir work/span_ner/fold0 --fold 0 --epochs 5 --threads 2 --max-span-width 64
```

Run long jobs in a dedicated tmux session, redirect stdout/stderr to a log under
`reports/span_ner/`, and record the session and PID. Before launch, require at least
20 GiB free on the project filesystem. Existing output directories are protected;
use `--resume` only with the identical configuration after inspecting failures.

The scores here measure character-span detection and negation only. They are not
the four competition scores. UMLS/HPO linking and patient association remain
separate stages. Old CPU, GPU, RAG and fusion artifacts remain comparison inputs.
