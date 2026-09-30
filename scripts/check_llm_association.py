"""Read-only progress and artifact checks for the association-review pilot."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main(plan_path):
    now = time.time()
    plan = json.loads(plan_path.read_text())
    report = ROOT / plan["report_dir"]
    queue = json.loads((report / "queue.json").read_text())
    folds = []
    for fold in plan["pilot_folds"]:
        directory = ROOT / plan["work_dir"] / f"fold{fold}"
        progress_path = directory / "progress.json"
        if not progress_path.exists():
            folds.append({"fold": fold, "status": "no_completed_task"})
            continue
        progress = json.loads(progress_path.read_text())
        recent = sorted((directory / "prompts").glob("*.json"), key=lambda p: p.stat().st_mtime)[-30:]
        costs = [json.loads(path.read_text())["elapsed_seconds"] for path in recent]
        seconds_per_task = sum(costs)/len(costs)
        remaining = progress["total"]-progress["completed"]
        summary = directory / "summary.json"
        status = "completed" if summary.exists() else "stopped" if queue["status"] == "failed" else "running"
        folds.append({**progress, "status": status,
                      "progress_age_seconds": now-progress_path.stat().st_mtime,
                      "recent_seconds_per_task": seconds_per_task,
                      "estimated_remaining_seconds": remaining*seconds_per_task,
                      "estimated_finish_utc": datetime.fromtimestamp(now+remaining*seconds_per_task, timezone.utc).isoformat()})
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "queue": queue, "folds": folds,
              "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).splitlines()}
    summary = report / "summary.json"
    if summary.exists():
        result["evaluation"] = json.loads(summary.read_text())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/llm_association/plan.json")
    main(parser.parse_args().plan)
