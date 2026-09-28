# Additional GPU 0 execution profile

Host: `dcf@10.253.27.177`, project `/home/dcf/chip2026`.
Spec: `experiments/span_ner/environment_gpu0.json`, canonical SHA256
`40ddda02cecef231196572735aaa226f72ba34d02a7037a89cfff776b51f6108`.

This profile reuses the already verified project virtual environment and model
bytes. No dependency, Python, driver or precision change is required. Only the
physical GPU binding changes from 2 to 0. Before acceptance, confirm GPU 0 has
no compute process; existing graphical processes are not stopped.

The following exact commands are saved with LF endings in
`scripts/check_gpu0_environment.sh` and copied to the same project-relative path
on the server. Verify the script equals this code block before execution.
On Windows do not pipe a PowerShell here-string into native SSH: the native pipe
can add CRLF, making the final script argument end in a literal CR. Preserve the
LF file bytes with SCP, then execute the saved script with this exact invocation:
`ssh -o BatchMode=yes 10.253.27.177 'bash /home/dcf/chip2026/scripts/check_gpu0_environment.sh'`.

Saved script contents for the seeded device witness:

```bash
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
```

Require exit 0 and the `CUDA_WITNESS` sentinel. Do not launch a training job as
part of this acceptance. The main task will use `SPAN_GPU_INDEX=0 bash
scripts/run_span_job.sh <run-name> <training-arguments>` only after acceptance;
the wrapper checks for compute processes again and takes a project GPU lock.
Other users' work is preserved. Logs, model checkpoints and data stay under
`/home/dcf/chip2026`.
