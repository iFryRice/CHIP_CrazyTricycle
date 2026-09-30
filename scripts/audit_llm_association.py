"""Explain relation removals after the declared pilot has finished scoring."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.association import _gold_concepts
from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.llm_association import should_veto


def run():
    plan = json.loads((ROOT / "experiments/llm_association/plan.json").read_text())
    report = ROOT / plan["report_dir"]
    summary = json.loads((report / "summary.json").read_text())
    if summary["plan_sha256"] != digest_file(ROOT / "experiments/llm_association/plan.json"):
        raise ValueError("Plan changed after evaluation.")
    documents = {str(document["pmc_id"]): document for document in read_jsonl(ROOT / plan["train_path"])}
    policies = {policy: {"removed_true": 0, "removed_false": 0, "by_patient_count": defaultdict(Counter),
                        "by_fold": defaultdict(Counter), "examples": []} for policy in plan["policies"]}
    format_failures, decisions_by_truth = [], defaultdict(Counter)
    for fold in plan["pilot_folds"]:
        directory = ROOT / plan["work_dir"] / f"fold{fold}"
        tasks = {task["task_id"]: task for task in json.loads((directory / "tasks.json").read_text())["tasks"]}
        records = json.loads((directory / "decisions.json").read_text())
        evidence = json.loads((directory / "summary.json").read_text())
        if evidence["decisions_sha256"] != digest_file(directory / "decisions.json"):
            raise ValueError("Decision file changed after inference.")
        for record in records:
            document = documents[record["pmc_id"]]
            gold = {relation["patient_id"]: _gold_concepts(relation["phenotype"]) for relation in document["association"]}
            truth = record["concept"] in gold[record["patient_id"]]
            outcome = "true" if truth else "false"
            decisions_by_truth[record["decision"]][outcome] += 1
            task = tasks[record["task_id"]]
            if record["parse_error"]:
                format_failures.append({"fold": fold, "task_id": record["task_id"], "error": record["parse_error"], "response": record["response"]})
            for policy, audit in policies.items():
                if not should_veto(record["decision"], record["complete_evidence"], policy):
                    continue
                audit[f"removed_{outcome}"] += 1
                audit["by_fold"][str(fold)][outcome] += 1
                audit["by_patient_count"]["single" if len(document["patient"]) == 1 else "multiple"][outcome] += 1
                audit["examples"].append({"fold": fold, "pmc_id": record["pmc_id"], "patient_id": record["patient_id"],
                    "concept": record["concept"], "label": task["concept_label"], "gold_positive": truth,
                    "baseline_score": task["baseline_score"], "complete_evidence": record["complete_evidence"],
                    "source_mentions": task["candidate_texts"], "evidence": record["evidence"]})
    for ranked in summary["ranked"]:
        audit = policies[ranked["policy"]]
        if summary["baseline_errors"]["tp"]-ranked["errors"]["tp"] != audit["removed_true"]:
            raise ValueError("Removed true relations do not reconcile with the scorer.")
        if summary["baseline_errors"]["fp"]-ranked["errors"]["fp"] != audit["removed_false"]:
            raise ValueError("Removed false relations do not reconcile with the scorer.")
    result = {"summary_sha256": digest_file(report / "summary.json"), "policies": policies,
              "decisions_by_truth": decisions_by_truth, "format_failures": format_failures,
              "scope": "Post-hoc diagnostic only. These labels were never given to the review model and cannot alter the frozen pilot gate."}
    write_json(report / "error_audit.json", result)
    print(json.dumps({policy: {key: value for key, value in audit.items() if key != "examples"} for policy, audit in policies.items()}))


if __name__ == "__main__":
    run()
