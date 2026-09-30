"""Run the frozen pilot after independent acceptance, then evaluate both folds."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/llm_extraction"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(state):
    path = REPORT / "pilot_queue.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(state), flush=True)


def main():
    started = time.time()
    plan_path = ROOT / "experiments/llm_extraction/plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["pilot_folds"] != [0, 1]:
        raise ValueError("This queue is dedicated to the frozen two-fold pilot.")
    if (REPORT / "pilot_queue.json").exists():
        raise FileExistsError("A previous queue exists; inspect its live processes before resuming.")
    state = {"status": "waiting_acceptance", "pid": os.getpid(), "started_at_epoch": started,
             "plan_sha256": digest(plan_path), "launcher_sha256": digest(Path(__file__)), "jobs": []}
    publish(state)
    acceptance_path = REPORT / "agent_acceptance.json"
    try:
        while not acceptance_path.exists():
            if time.time() - started > 7200:
                raise TimeoutError("Independent runtime acceptance did not complete within two hours.")
            if not (REPORT / "environment_smoke.json").exists():
                alive = subprocess.run(["tmux", "has-session", "-t", "chip-qwen3-finalize"], capture_output=True).returncode == 0
                if not alive:
                    raise RuntimeError("Model staging/smoke is terminal without a passing witness.")
            time.sleep(10)
        spec = json.loads((ROOT / "experiments/llm_extraction/environment.json").read_text())
        spec_sha = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for name in ["environment_smoke.json", "environment_acceptance.json", "agent_acceptance.json"]:
            evidence = json.loads((REPORT / name).read_text())
            if evidence["status"] != "passed" or evidence["spec_sha256"] != spec_sha:
                raise ValueError(f"Runtime acceptance failed or belongs to another specification: {name}")
        independent = json.loads(acceptance_path.read_text())
        if independent["witness_report_sha256"] != digest(REPORT / "environment_acceptance.json"):
            raise ValueError("The fresh agent's witness report changed after verification.")
        if digest(plan_path) != state["plan_sha256"]:
            raise ValueError("The frozen pilot changed while the queue was waiting.")
        children = []
        state["status"] = "inference_running"
        for fold, gpu in [(0, 2), (1, 0)]:
            log = REPORT / f"fold{fold}_inference.log"
            handle = log.open("x", encoding="utf-8")
            env = {**os.environ, "LLM_GPU_INDEX": str(gpu)}
            command = ["bash", "scripts/run_llm_extraction.sh", "--fold", str(fold)]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            handle.close()
            children.append(process)
            state["jobs"].append({"fold": fold, "physical_gpu": gpu, "pid": process.pid, "command": command,
                                  "log": str(log.relative_to(ROOT)), "status": "running"})
        publish(state)
        while any(process.poll() is None for process in children):
            for process, job in zip(children, state["jobs"], strict=True):
                if process.poll() is not None and job["status"] == "running":
                    job.update(status="completed" if process.returncode == 0 else "failed", exit_code=process.returncode)
                    publish(state)
            time.sleep(10)
        for process, job in zip(children, state["jobs"], strict=True):
            job.update(status="completed" if process.returncode == 0 else "failed", exit_code=process.returncode)
        if any(process.returncode != 0 for process in children):
            raise RuntimeError("A pilot inference process failed; inspect its preserved log and prompt cache.")
        state["status"] = "evaluating"
        publish(state)
        with (REPORT / "pilot_evaluation.log").open("x", encoding="utf-8") as handle:
            result = subprocess.run(["bash", "scripts/run_llm_extraction.sh", "evaluate"], cwd=ROOT,
                                    env={**os.environ, "LLM_GPU_INDEX": "2"}, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError("Pilot evaluation failed; inspect the preserved evaluation log.")
        summary = json.loads((ROOT / plan["report_dir"] / "summary.json").read_text())
        state.update(status="completed", selected=summary["selected"], best_mean_gain=summary["ranked"][0]["mean_gain"],
                     elapsed_seconds=time.time() - started)
        publish(state)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.time() - started)
        publish(state)
        raise


if __name__ == "__main__":
    main()
