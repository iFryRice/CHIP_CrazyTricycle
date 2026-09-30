"""Review the same source windows with explicit provided patient-role markers."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.llm_patient_markers import mark_task
from scripts.run_llm_association import load_plan, prepare
from scripts.run_constrained_association import infer, evaluate_pilot


def prepare_marked(plan_path, plan):
    prepare(plan_path, plan)
    patients = {str(document["pmc_id"]): document["patient"] for document in read_jsonl(ROOT / plan["train_path"])}
    packs = []
    for fold in plan["pilot_folds"]:
        directory = ROOT / plan["work_dir"] / f"fold{fold}"
        pack = json.loads((directory / "tasks.json").read_text())
        pack["tasks"] = [mark_task(task, patients[task["pmc_id"]]) for task in pack["tasks"]]
        packs.append((directory, pack))
    for directory, pack in packs:
        write_json(directory / "tasks.json", pack)
        manifest = json.loads((directory / "prepared.json").read_text())
        manifest.update(tasks_sha256=digest_file(directory / "tasks.json"), patient_marker_protocol="provided_anchor_labels",
                        original_text_preserved=True, marker_source="Given patient references only; no phenotype or association labels.")
        write_json(directory / "prepared.json", manifest)


def check_marked(plan, fold):
    directory = ROOT / plan["work_dir"] / f"fold{fold}"
    manifest = json.loads((directory / "prepared.json").read_text())
    pack = json.loads((directory / "tasks.json").read_text())
    if manifest.get("patient_marker_protocol") != "provided_anchor_labels" or any(task.get("patient_marker_protocol") != "provided_anchor_labels" for task in pack["tasks"]):
        raise ValueError("Patient-role annotation preparation is incomplete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "preflight", "infer", "evaluate"])
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/llm_patient_markers/plan.json")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.stage == "prepare":
        prepare_marked(args.plan, plan)
    elif args.stage == "evaluate":
        for fold in plan["pilot_folds"]:
            check_marked(plan, fold)
        evaluate_pilot(args.plan, plan)
    else:
        check_marked(plan, args.fold)
        infer(args.plan, plan, args.fold, preflight=args.stage == "preflight")
