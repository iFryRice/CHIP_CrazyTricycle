#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
.venv/bin/python -u scripts/stage_qwen_model.py --request work/span_ner/qwen3_download/request.json --workers 48 2>&1 | tee -a reports/llm_extraction/model_download.log
