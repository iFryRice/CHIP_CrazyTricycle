# SSH environment ledger

Provider: `dcf@10.253.27.177`; project `/home/dcf/chip2026`.

## span-ner-gpu0@40ddda02

- Spec: `experiments/span_ner/environment_gpu0.json`
- Canonical spec SHA256: `40ddda02cecef231196572735aaa226f72ba34d02a7037a89cfff776b51f6108`
- Same project-local virtual environment, locked packages and verified model bytes as `span-ner@1327ca93`; only physical device binding differs.
- Tier: GPU 0, Tesla V100 32 GiB, 2 CPU threads, FP16.
- Status: passed fresh exact-runbook acceptance, exit 0 and CUDA_WITNESS; Python 3.12.13, torch 2.6.0+cu124, transformers 4.49.0, V100 capability [7,0], seeded FP16 loss 257.83489990234375 and peak 16.51611328125 MiB.
- Windows transport correction: native PowerShell stdin appended CR to the final shell argument. The documented LF-only script (519 bytes, verified identical to the code block) is copied with SCP and invoked by SSH, preserving exact command bytes.
- Invocation: `reports/span_ner/gpu0_runbook.md`; wrapper rechecks for other compute processes and holds a per-GPU project lock.

## span-ner@1327ca93

- Spec: `experiments/span_ner/environment.json`
- Canonical spec SHA256: `1327ca93670a7ea2c01e0ee41acc3af11c736c3a622a0079480e8ea1e88aa8ba`
- Shape: direct SSH host, project-local `.venv` and `.tools/uv`
- Build: `.tools/uv sync --locked --extra training --python 3.12`
- Runtime: Python 3.12.13; torch 2.6.0+cu124; transformers 4.49.0
- Tier: GPU 2, Tesla V100 32 GiB; 2 CPU threads; FP16
- Weights: `models/biomedbert`, pinned official revision in spec; all six files independently verified
- Validation status: passed independent exact-runbook acceptance, all 48 tests without skips, seeded CUDA FP16 forward/backward witness, 20-step smoke and the complete five-epoch fold-0 run
- Evidence: `reports/span_ner/environment_acceptance.json`, `smoke_fold0/summary.json`, `fold0/summary.json` and `fold0/training.jsonl`
- Formal run: completed 2026-09-28 22:40:38 Asia/Shanghai; 490.209 seconds, 692 applied optimizer updates, 3 AMP overflow skips in epoch 1, peak allocated GPU memory 3.05750 GiB
- Outcome boundary: runtime acceptance passed; default-threshold span precision remains low. HPO normalization and patient association are not included in this training run.
- Reproduction: `reports/span_ner/remote_runbook.md`

Known setup details:
- System Python 3.8 is untouched. Python 3.12 is managed under `.tools/python`.
- uv 0.11.19 came from its official PyPI Linux wheel after GitHub transfers timed out; wheel SHA256 was verified before extraction.
- The host rejects huggingface.co connections. The pinned official redirect was resolved on the local computer; the host downloaded from the same official CDN. Official Git/LFS hashes all matched.
- Memory cache is reclaimable; monitor MemAvailable and disk space, not free RAM alone.
- The large PyTorch wheel was limited to roughly 0.1 MB/s per connection. `scripts/download_verified_file.py` retrieved the same official wheel with strict HTTP ranges, and the final SHA256 matched `uv.lock`: `a393b506844035c0dac2f30ea8478c343b8e95a429f06f3b3cadfc7f53adb597`.
- The original installer alone was stopped after that verification. The identical named package was preloaded with `.tools/uv pip install --python .venv/bin/python --no-deps --no-index --find-links work/span_ner/wheelhouse 'torch==2.6.0+cu124'`, followed by the unchanged locked bootstrap using cached dependencies with `UV_OFFLINE=1`.
