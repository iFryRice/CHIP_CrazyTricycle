"""Bounded two-device queue for a frozen local LLM review experiment."""

import argparse
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


def run(plan_path):
    plan = load_plan(plan_path)
    report = ROOT / plan["report_dir"]
    report.mkdir(parents=True, exist_ok=True)
    state_path = report / "queue.json"
    if state_path.exists():
        raise FileExistsError("Inspect previous queue state before resuming.")
    runner, wrapper = plan["runner_script"], plan["wrapper_script"]
    if runner not in plan["code_sha256"] or wrapper not in plan["code_sha256"]:
        raise ValueError("Queue entrypoints must be frozen in the plan.")
    started = time.time()
    state = {"status": "starting", "pid": os.getpid(), "started_at_epoch": started,
             "plan_sha256": digest_file(plan_path), "jobs": []}
    atomic_json(state_path, state)
    pending, active, failed = list(plan["pilot_folds"]), {}, False
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1"}
    try:
        while pending or active:
            for gpu in [2, 0]:
                if gpu in active or not pending or failed:
                    continue
                fold = pending.pop(0)
                log = report / f"fold{fold}_inference.log"
                command = ["bash", wrapper, "--plan", str(plan_path), "--fold", str(fold)]
                with log.open("x", encoding="utf-8") as handle:
                    process = subprocess.Popen(command, cwd=ROOT, env={**env, "LLM_GPU_INDEX": str(gpu)}, stdout=handle, stderr=subprocess.STDOUT)
                job = {"fold": fold, "gpu": gpu, "pid": process.pid, "log": str(log.relative_to(ROOT)), "status": "running"}
                state["jobs"].append(job)
                active[gpu] = process, job
            state["status"] = "inference_running"
            for gpu, (process, job) in list(active.items()):
                code = process.poll()
                if code is not None:
                    job.update(exit_code=code, status="completed" if code == 0 else "failed")
                    failed |= code != 0
                    del active[gpu]
            atomic_json(state_path, state)
            if failed and not active:
                raise RuntimeError("Inference failed; logs and caches are preserved. No pending folds were launched.")
            if pending or active:
                time.sleep(10)
        state["status"] = "evaluating"
        atomic_json(state_path, state)
        with (report / "evaluation.log").open("x", encoding="utf-8") as handle:
            subprocess.run([str(ROOT / ".venv/bin/python"), runner, "evaluate", "--plan", str(plan_path)],
                           cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((report / "summary.json").read_text())
        state.update(status="completed", selected=summary["selected"], best_mean_gain=summary["ranked"][0]["mean_gain"])
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.time()-started
        atomic_json(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    run(parser.parse_args().plan.resolve())
