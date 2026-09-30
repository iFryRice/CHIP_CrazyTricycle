"""Run two declared association-review folds and preserve all terminal states."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_llm_association import atomic_json, load_plan
from patientphex.data import digest_file


def main():
    plan_path = ROOT / "experiments/llm_association/plan.json"
    plan = load_plan(plan_path)
    report = ROOT / plan["report_dir"]
    report.mkdir(parents=True, exist_ok=True)
    state_path = report / "queue.json"
    if state_path.exists():
        raise FileExistsError("Inspect prior queue state before starting another invocation.")
    started = time.time()
    state = {"status": "starting", "pid": os.getpid(), "started_at_epoch": started,
             "plan_sha256": digest_file(plan_path), "jobs": []}
    atomic_json(state_path, state)
    children = []
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1"}
    try:
        for fold, gpu in [(0, 2), (1, 0)]:
            log = report / f"fold{fold}_inference.log"
            with log.open("x", encoding="utf-8") as handle:
                process = subprocess.Popen(["bash", "scripts/run_llm_association.sh", "--fold", str(fold)],
                    cwd=ROOT, env={**env, "LLM_GPU_INDEX": str(gpu)}, stdout=handle, stderr=subprocess.STDOUT)
            children.append(process)
            state["jobs"].append({"fold": fold, "gpu": gpu, "pid": process.pid, "log": str(log.relative_to(ROOT))})
        state["status"] = "inference_running"
        atomic_json(state_path, state)
        for process, job in zip(children, state["jobs"], strict=True):
            job["exit_code"] = process.wait()
            atomic_json(state_path, state)
        if any(process.returncode != 0 for process in children):
            raise RuntimeError("Association inference failed; logs and prompt caches are preserved.")
        state["status"] = "evaluating"
        atomic_json(state_path, state)
        with (report / "evaluation.log").open("x", encoding="utf-8") as handle:
            subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/run_llm_association.py", "evaluate"],
                           cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((report / "summary.json").read_text())
        state.update(status="completed", selected=summary["selected"], best_mean_gain=summary["ranked"][0]["mean_gain"])
    except Exception as error:
        # Already running children retain their output; never obscure a partial launch.
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.time()-started
        atomic_json(state_path, state)


if __name__ == "__main__":
    main()
