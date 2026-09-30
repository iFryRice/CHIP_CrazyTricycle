#!/usr/bin/env bash
set -euo pipefail
cd /home/dcf/chip2026
export UV_CACHE_DIR=/home/dcf/chip2026/.uv-cache
export UV_PYTHON_INSTALL_DIR=/home/dcf/chip2026/.tools/python
export UV_PROJECT_ENVIRONMENT=/home/dcf/chip2026/experiments/llm_runtime/.venv
export UV_HTTP_TIMEOUT=180
export UV_CONCURRENT_DOWNLOADS=4
.venv/bin/python - <<'PY'
import hashlib, json, tomllib
from pathlib import Path
spec = json.loads(Path('experiments/llm_extraction/environment.json').read_text())
for path, key in [('experiments/llm_runtime/pyproject.toml', 'pyproject_sha256'), ('experiments/llm_runtime/uv.lock', 'lock_sha256'), ('experiments/llm_extraction/model.json', 'model_manifest_sha256')]:
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != spec[key]:
        raise ValueError(f'Environment input changed: {path}')
print('Verified declared environment inputs.')
wheel = Path('work/span_ner/wheelhouse/torch-2.6.0+cu124-cp312-cp312-linux_x86_64.whl')
lock = tomllib.loads(Path('experiments/llm_runtime/uv.lock').read_text())
package = next(item for item in lock['package'] if item['name'] == 'torch')
expected = {item['hash'] for item in package['wheels'] if 'cp312' in item['url'] and 'linux_x86_64' in item['url']}
digest = hashlib.sha256()
with wheel.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
if 'sha256:' + digest.hexdigest() not in expected:
    raise ValueError('Cached PyTorch wheel differs from the frozen lock.')
print('Verified cached PyTorch wheel against uv.lock.')
for name in ['transformers', 'peft', 'accelerate', 'psutil']:
    package = next(item for item in lock['package'] if item['name'] == name)
    candidates = [item for item in package['wheels'] if item['url'].endswith('-py3-none-any.whl') or 'cp36-abi3-manylinux2010_x86_64' in item['url']]
    if len(candidates) != 1:
        raise ValueError(f'Ambiguous staged wheel: {name}')
    item = candidates[0]
    path = Path('work/span_ner/wheelhouse') / item['url'].split('/')[-1]
    if path.stat().st_size != item['size'] or 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest() != item['hash']:
        raise ValueError(f'Staged dependency differs from uv.lock: {name}')
print('Verified staged runtime wheels against uv.lock.')
PY
if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
    .tools/uv venv --python /home/dcf/chip2026/.venv/bin/python "$UV_PROJECT_ENVIRONMENT"
fi
.tools/uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps --no-index --find-links work/span_ner/wheelhouse 'torch==2.6.0+cu124' 'transformers==4.51.3' 'peft==0.15.2' 'accelerate==1.6.0' 'psutil==7.2.2'
.tools/uv sync --offline --locked --project experiments/llm_runtime --python /home/dcf/chip2026/.venv/bin/python
