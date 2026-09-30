# Qwen3 runtime acceptance

Provider: `dcf@10.253.27.177`. Project: `/home/dcf/chip2026`.

Declared spec: `experiments/llm_extraction/environment.json`.
Canonical SHA-256: `e9dd0d2face037abbbadeef7478721df212680749dbc6de7b50193d01a16a1a3`.
Model: official `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.

This runtime is isolated in `experiments/llm_runtime/.venv`. Dependencies are governed by that subproject's `pyproject.toml` and `uv.lock`. All actual files are under `/home/dcf/chip2026`; no global packages are modified.

The provider already has the exact locked PyTorch wheel at `work/span_ner/wheelhouse/torch-2.6.0+cu124-cp312-cp312-linux_x86_64.whl`. The bootstrap checks its SHA-256 against `uv.lock`, preloads the named package through uv, then runs the same locked sync. This avoids redownloading the identical 768 MB wheel under another cache origin; it does not change any dependency or package bytes.

Direct provider downloads timed out even with a 180-second timeout. The exact locked transformers, peft, accelerate and psutil wheels are therefore also staged in that wheelhouse through the workstation's official PyPI connection and SSH. Bootstrap verifies each against the same lock, preloads them with uv, and completes locked sync offline using the provider's existing dependency cache. A missing or mismatched wheel fails explicitly; no alternate dependency version is selected.

After the pinned model download has produced `reports/llm_extraction/model_verification.json`, build the declared runtime:

```bash
cd /home/dcf/chip2026
bash scripts/bootstrap_qwen_runtime.sh
```

Run acceptance exactly as follows, using the project wrapper and its GPU 2 lock:

```bash
cd /home/dcf/chip2026
bash scripts/run_qwen_witness.sh --report reports/llm_extraction/environment_acceptance.json
```

The command must exit 0, print `QWEN_CUDA_WITNESS`, and save a report with `status=passed`, matching spec SHA, successful finite LoRA optimizer update, actual model parameter count 8190735360, and exact-source JSON generation. It checks every model file's content hash before loading. Imports alone are insufficient.

If another compute process owns GPU 2, the wrapper exits without taking the device. Inspect that process; do not relaunch blindly or use another user's device. GPU 0 is also declared and can be selected explicitly with `QWEN_GPU_INDEX=0`; record the selected physical GPU in the acceptance evidence.

The seeded small Qwen3 model validates FP16 LoRA training kernels. The full 8.19B model validates actual FP16 loading and generation. Neither witness is a PatientPheX accuracy result. A fresh agent must follow this runbook and report any documentation/runtime divergence before formal extraction experiments are launched.
