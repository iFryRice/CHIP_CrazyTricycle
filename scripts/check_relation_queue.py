"""Read-only compact status and ETA for relation training queues."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file


def status(plan_path):
    plan = json.loads(plan_path.read_text())
    report = ROOT / plan["report_dir"]
    queue = json.loads((report / "queue.json").read_text())
    now = datetime.now(timezone.utc)
    result = {"checked_at_utc": now.isoformat(), "status": queue["status"],
              "plan_matches_queue": digest_file(plan_path) == queue["plan_sha256"],
              "completed_jobs": sum(job["status"] == "completed" for job in queue["jobs"]),
              "failed_jobs": sum(job["status"] == "failed" for job in queue["jobs"]),
              "pending_jobs": queue.get("pending_jobs", 0), "active": []}
    durations, slots = [], {0: 0.0, 2: 0.0}
    for job in queue["jobs"]:
        work_dir = plan["members"][str(job["seed"])]["work_dir"] if "seed" in job else plan["work_dir"]
        work = ROOT / work_dir / f"fold{job['fold']}"
        if (work / "summary.json").exists():
            summary = json.loads((work / "summary.json").read_text())
            durations.append(summary["elapsed_seconds"]+15)
        if job["status"] != "running": continue
        row = {key: job[key] for key in ["fold", "gpu", "pid"]}
        if "seed" in job: row["seed"] = job["seed"]
        process = Path(f"/proc/{job['pid']}")
        try:
            row["worker_verified_alive"] = (process / "cwd").resolve() == ROOT and "run_relation_text.sh" in (process / "cmdline").read_text().replace("\x00", " ")
        except (FileNotFoundError, PermissionError):
            row["worker_verified_alive"] = False
        remaining = 175.0
        log = work / "training.jsonl"
        if log.exists():
            records = []
            for line in log.read_text().splitlines():
                try: records.append(json.loads(line))
                except json.JSONDecodeError: break
            if records:
                row["latest_event"] = records[-1]["event"]
                steps = [event for event in records if event["event"] == "train_step"]
                if records[-1]["event"] == "complete":
                    remaining = 0.0
                elif len(steps) >= 2:
                    first, last = steps[max(0, len(steps)-4)], steps[-1]
                    rate = (last["elapsed_seconds"]-first["elapsed_seconds"])/max(1,last["step"]-first["step"])
                    row.update(step=last["step"], total_steps=last["planned_steps"], epoch=last["epoch"], amp_scale=last["amp_scale"])
                    remaining = rate*(last["planned_steps"]-last["step"])+15
                else:
                    remaining = max(15.0, 175-records[-1]["elapsed_seconds"])
        row["estimated_seconds_remaining"] = round(remaining,1)
        slots[job["gpu"]] = remaining
        result["active"].append(row)
    mean_duration = sum(durations)/len(durations) if durations else 175.0
    for _ in range(result["pending_jobs"]):
        gpu = min(slots, key=slots.get)
        slots[gpu] += mean_duration
    if queue["status"] in {"starting", "training"}:
        remaining = max(slots.values())+20
        result["estimated_total_seconds_remaining"] = round(remaining,1)
        result["estimated_finish_utc"] = (now+timedelta(seconds=remaining)).isoformat()
    if "error" in queue: result["error"] = queue["error"]
    if (report / "summary.json").exists():
        summary = json.loads((report / "summary.json").read_text())
        result["evaluation"] = {key:summary[key] for key in ["promoted","checks","fold_deltas","confirmation_mean_gain"] if key in summary}
        if "metrics" in summary: result["evaluation"]["score"] = summary["metrics"]["score"]
    result["gpu"] = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader"], text=True).splitlines()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    status(parser.parse_args().plan.resolve())
