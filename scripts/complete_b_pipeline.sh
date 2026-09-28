#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0

# Wait only for the two already-started confirmation jobs; never relaunch them.
/usr/bin/python3 - <<'PY'
import json, pathlib, time
deadline = time.monotonic() + 20 * 60
previous = None
while True:
    states = []
    for fold in (3, 4):
        path = pathlib.Path('work/span_ner/boundary-cap100-fold%d/summary.json' % fold)
        result = json.loads(path.read_text()) if path.exists() else {'status': 'initializing'}
        if result['status'] == 'failed':
            raise SystemExit('Confirmation fold %d failed; inspect its existing log' % fold)
        states.append((fold, result['status'], result.get('epochs_completed', 0)))
    if states != previous:
        print(json.dumps({'waiting_for_existing_jobs': states}), flush=True)
        previous = states
    if all(status == 'completed' and epochs == 5 for _, status, epochs in states):
        break
    if time.monotonic() > deadline:
        raise SystemExit('Timed out waiting for existing confirmation jobs; no job was restarted')
    time.sleep(20)
PY

for fold in 3 4; do
  .tools/uv run --no-sync python scripts/evaluate_linked_pipeline.py \
    --run-dir "work/span_ner/boundary-cap100-fold${fold}" \
    --umls-terms resources/umls_hpo_terms.tsv.gz \
    --frozen-selection experiments/span_ner/selected_pipeline.json \
    --output-dir "reports/span_ner/confirmed_boundary-cap100-fold${fold}" \
    2>&1 | tee "reports/span_ner/confirmed_fold${fold}.log"
done

/usr/bin/python3 - <<'PY'
import json, pathlib
choice = json.load(open('experiments/span_ner/selected_pipeline.json'))['selected']
for fold in range(1, 5):
    directory = pathlib.Path('reports/span_ner/confirmed_boundary-cap100-fold%d' % fold)
    result = json.loads((directory/'comparison.json').read_text())
    assert result['frozen_selection'] is True
    for key in ('entity_strategy', 'association_threshold', 'span_threshold'):
        assert result['selected'][key] == choice[key]
    assert json.loads((directory/'validation_report.json').read_text())['valid']
print('All four confirmation folds completed with the same frozen pipeline', flush=True)
PY

SPAN_GPU_INDEX=0 bash scripts/run_span_job.sh final-cap100-all80 \
  --fit-all --epochs 5 --pos-weight-cap 100

.tools/uv run --no-sync python scripts/predict_span_ner.py \
  --run-dir work/span_ner/final-cap100-all80 --checkpoint last.pt \
  --data PatientPheX-V1-B/PatientPheX-V1-B.jsonl --role target --score-floor 0.5 \
  --output-dir work/span_ner/b_span_predictions

.tools/uv run --no-sync python scripts/finalize_b_pipeline.py \
  --selection experiments/span_ner/selected_pipeline.json \
  --ner-run work/span_ner/final-cap100-all80 \
  --spans work/span_ner/b_span_predictions/spans.jsonl \
  --umls-terms resources/umls_hpo_terms.tsv.gz \
  --output submissions/patientphex_b.jsonl --report-dir reports/b_supervised \
  --model-dir work/span_ner/b_linking
