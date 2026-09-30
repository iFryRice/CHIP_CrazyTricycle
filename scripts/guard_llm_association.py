"""Stop only this pilot's verified workers if its format gate becomes impossible."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file
from scripts.run_llm_association import atomic_json, load_plan


def process(pid):
    directory = Path(f"/proc/{pid}")
    if not directory.exists():
        return None
    return {"pid": pid, "command": (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
            "parent": int((directory / "stat").read_text().rsplit(")", 1)[1].split()[1]),
            "cwd": str((directory / "cwd").resolve()), "uid": directory.stat().st_uid}


def main(plan_path):
    plan_path = plan_path.resolve()
    plan = load_plan(plan_path)
    report = ROOT / plan["report_dir"]
    while True:
        queue = json.loads((report / "queue.json").read_text())
        if queue["status"] in {"completed", "failed"}:
            return
        progress = []
        impossible = False
        for fold in plan["pilot_folds"]:
            directory = ROOT / plan["work_dir"] / f"fold{fold}"
            if not (directory / "progress.json").exists():
                continue
            current = json.loads((directory / "progress.json").read_text())
            prepared = json.loads((directory / "prepared.json").read_text())
            if current["total"] != prepared["task_count"] or prepared["plan_sha256"] != digest_file(plan_path):
                raise ValueError("The guard does not match the frozen pilot.")
            current["maximum_possible_format_success_rate"] = 1-current["format_errors"]/current["total"]
            impossible |= current["maximum_possible_format_success_rate"] < plan["promotion"]["minimum_format_success_rate"]
            progress.append(current)
        if impossible:
            if queue["plan_sha256"] != digest_file(plan_path):
                raise ValueError("Queue identity changed.")
            candidates = []
            for job in queue["jobs"]:
                parent = process(job["pid"])
                if parent is None:
                    continue
                wrapper = plan.get("wrapper_script", "scripts/run_llm_association.sh")
                runner = plan.get("runner_script", "scripts/run_llm_association.py")
                arguments = (["--plan", str(plan_path)] if "runner_script" in plan else []) + ["--fold", str(job["fold"])]
                if parent["cwd"] != str(ROOT) or parent["command"].split() != ["bash", wrapper, *arguments]:
                    raise ValueError("Refusing to signal a different wrapper process.")
                for directory in Path("/proc").iterdir():
                    if not directory.name.isdigit():
                        continue
                    try:
                        child = process(int(directory.name))
                    except (FileNotFoundError, PermissionError, ProcessLookupError):
                        continue
                    if child is None or child["parent"] != job["pid"]:
                        continue
                    command = child["command"].split()
                    expected = ["-u", runner, "infer", *arguments]
                    if command[1:] != expected or child["cwd"] != str(ROOT) or child["uid"] != os.getuid():
                        raise ValueError("Refusing to signal an unrelated child process.")
                    candidates.append(child)
            evidence = {"status": "format_gate_impossible", "plan_sha256": digest_file(plan_path), "progress": progress,
                        "minimum_format_success_rate": plan["promotion"]["minimum_format_success_rate"],
                        "reason": "Even if every remaining response were valid, one fold cannot reach the preregistered format floor.",
                        "stopped_workers": candidates, "accuracy_evaluated": False, "time_epoch": time.time()}
            atomic_json(report / "futility_stop.json", evidence)
            for worker in candidates:
                latest = process(worker["pid"])
                if latest != worker:
                    raise ValueError("Worker identity changed before termination.")
                os.kill(worker["pid"], signal.SIGTERM)
            return
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/llm_association/plan.json")
    main(parser.parse_args().plan)
