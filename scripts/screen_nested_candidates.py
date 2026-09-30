"""Two-fold entity-only screen; do not interpret unchanged associations as a full pipeline."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.evaluation import evaluate
from patientphex.overlap_candidates import add_nested
from scripts.optimize_cpu_association import blind
from scripts.run_llm_association import load_plan


def entity_metrics(source, predictions):
    metrics = evaluate(source, predictions)
    return {key: metrics[key] for key in ["mention", "document", "counts"]}


def run(path):
    plan = load_plan(path)
    report = ROOT / plan["report_dir"]
    if (report / "summary.json").exists():
        raise FileExistsError("Preserve an earlier entity screen.")
    documents = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["train_path"])}
    original = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["before_review"])}
    baseline = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["baseline_oof"])}
    folds = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    source, previous, predictions, fold_rows = [], [], [], []
    for fold in plan["pilot_folds"]:
        candidates = json.loads((ROOT / plan["candidates"][str(fold)]).read_text())
        if set(candidates) != set(folds[fold]):
            raise ValueError("Candidate source is not the declared held-out fold.")
        local, changes = [], []
        validation = [documents[i] for i in folds[fold]]
        old = [baseline[i] for i in folds[fold]]
        for document, previous_record in zip(validation, old, strict=True):
            identifier = str(document["pmc_id"])
            entities, added = add_nested(blind(document), original[identifier]["entities"], previous_record["entities"],
                                          candidates[identifier], **plan["fixed_thresholds"])
            local.append({**previous_record, "entities": entities})
            changes.extend({"pmc_id": identifier, "entity": entity} for entity in added)
        row = {"fold": fold, "baseline": entity_metrics(validation, old), "candidate": entity_metrics(validation, local), "added": len(changes)}
        fold_rows.append(row)
        source.extend(validation); previous.extend(old); predictions.extend(local)
        write_json(report / f"fold{fold}_edits.json", changes)
    old, new = entity_metrics(source, previous), entity_metrics(source, predictions)
    delta = new["mention"]["f1"]-old["mention"]["f1"]
    checks = {"mention_gain": delta >= plan["screen_gates"]["minimum_mention_f1_gain"],
              "document_guard": new["document"]["f1"] >= old["document"]["f1"]-plan["screen_gates"]["maximum_document_f1_loss"],
              "fold_guard": all(row["candidate"]["mention"]["f1"] >= row["baseline"]["mention"]["f1"] for row in fold_rows)}
    result = {"plan_sha256": digest_file(path), "baseline": old, "candidate": new, "folds": fold_rows,
              "checks": checks, "advance_to_joint_test": all(checks.values()), "mention_f1_gain": delta,
              "scope": "Entity-only screening on reused development folds 0/1. Associations were not recomputed, so no full score or association performance is reported. This stage cannot authorize B generation."}
    write_json(report / "summary.json", result)
    write_json(report / "plan.json", plan)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", type=Path, required=True)
    run(parser.parse_args().plan.resolve())
