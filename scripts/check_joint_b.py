"""Read-only evidence of the live joint B controller and its subprocesses."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def process_evidence(pid, marker):
    path = Path("/proc") / str(pid)
    try:
        argv = (path / "cmdline").read_bytes().decode().rstrip("\x00").split("\x00")
        cwd = (path / "cwd").resolve(strict=True)
        children_path = path / "task" / str(pid) / "children"
        children = [int(value) for value in children_path.read_text().split()]
        child_commands = []
        for child in children:
            try:
                child_commands.append({"pid": child, "argv": (Path("/proc") / str(child) / "cmdline").read_bytes().decode().rstrip("\x00").split("\x00")})
            except FileNotFoundError:
                child_commands.append({"pid": child, "gone_during_read": True})
        return {"pid": pid, "verified_live": cwd == ROOT and any(marker in value for value in argv),
                "cwd": str(cwd), "argv": argv, "children": child_commands}
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": pid, "verified_live": False, "process_missing": True}
    except PermissionError:
        return {"pid": pid, "verified_live": None, "permission_denied": True}


def run(plan_path):
    plan = read(plan_path)
    report = ROOT / plan["report_dir"]
    state = read(report / "queue.json")
    expected = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if state["plan_sha256"] != expected:
        raise ValueError("The observed queue belongs to a different frozen B plan.")
    concept = read(ROOT / plan["concept_plan"])
    now = time.time()
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": state["status"],
              "plan_sha256": expected, "controller": process_evidence(state["pid"], "run_joint_b.py"),
              "live_jobs": [], "concept_shards": []}
    for job in state["jobs"]:
        if job["status"] == "running":
            marker = "run_concept_review.sh" if job["phase"] == "concept" else "run_relation_b.sh"
            result["live_jobs"].append({**job, "process": process_evidence(job["pid"], marker)})
    estimates = []
    for shard in concept["pilot_folds"]:
        path = ROOT / concept["work_dir"] / f"fold{shard}" / "progress.json"
        if path.exists():
            progress = read(path)
            remaining = progress["total"] - progress["completed"]
            estimate = progress["elapsed_seconds"] * remaining / progress["completed"] if progress["completed"] else None
            estimates.append(estimate)
            result["concept_shards"].append({"shard": shard, **progress, "progress_age_seconds": now-path.stat().st_mtime,
                                             "estimated_remaining_seconds": estimate})
    if estimates and all(value is not None for value in estimates):
        result["concept_stage_estimated_remaining_seconds"] = max(estimates)
        result["eta_scope"] = "Concept stage only, extrapolated from observed average throughput; not a guarantee or full pipeline ETA."
    path = report / "progress.json"
    if path.exists():
        result["association_progress"] = read(path)
        result["association_progress_age_seconds"] = now-path.stat().st_mtime
        result["completed_text_members"] = len(list((ROOT / plan["work_dir"]).glob("seed_*_probabilities.json")))
        result["total_text_members"] = len(plan["text_models"])
    path = report / "summary.json"
    if path.exists():
        summary = read(path)
        output = ROOT / plan["output"]
        result["candidate"] = {"status": summary["status"], "output": plan["output"], "sha256": summary["sha256"],
                               "validated": summary["validation"]["valid"],
                               "output_hash_matches": output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() == summary["sha256"]}
    if state.get("error"):
        result["error"] = state["error"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/joint_b/plan.json")
    run(parser.parse_args().plan.resolve())
