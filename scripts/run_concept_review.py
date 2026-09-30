"""Prepare and evaluate contextual HPO meaning checks on frozen OOF entities."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.abbreviations import extract_definitions, short_form
from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.llm_concept_review import build_task, eligible, ontology_definitions, parse_decision
from patientphex.ontology import Ontology
from patientphex.patient_linking import _check_provenance
from patientphex.span_linking import SpanLinker
from scripts.infer_structured_review import infer
from scripts.run_llm_association import load_plan, load_tasks
from scripts.optimize_cpu_association import blind, bootstrap_delta, error_counts, from_scores, validate_file


def inputs(plan):
    documents = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["train_path"])}
    baseline = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["baseline_oof"])}
    folds = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    if sorted(i for values in folds for i in values) != sorted(documents) or set(documents) != set(baseline):
        raise ValueError("Source, baseline or fold coverage differs.")
    return documents, baseline, folds


def association_model(plan, fold, documents, validation_ids):
    with (ROOT / plan["association_models"][str(fold)]).open("rb") as handle:
        model = pickle.load(handle)
    _check_provenance(model.provenance_, set(documents)-set(validation_ids), set(validation_ids))
    return model


def prepare(plan_path, plan):
    documents, baseline, folds = inputs(plan)
    ontology = Ontology(ROOT / plan["ontology_path"])
    fixed, definitions = SpanLinker(ontology), ontology_definitions(ROOT / plan["ontology_path"])
    root = ROOT / plan["work_dir"]
    if root.exists():
        raise FileExistsError("Do not overwrite another concept-review experiment.")
    packs = []
    for fold in plan["pilot_folds"]:
        model = association_model(plan, fold, documents, folds[fold])
        tasks = []
        for identifier in folds[fold]:
            document = blind(documents[identifier])
            entities = baseline[identifier]["entities"]
            scores = model.predict_scores(document, entities)
            if from_scores(document, entities, scores, 0.5) != baseline[identifier]:
                raise ValueError("Context-linker baseline failed exact replay.")
            local_definitions = extract_definitions(document)
            tasks.extend(build_task(document, entity, index, ontology, definitions, local_definitions)
                         for index, entity in enumerate(entities) if eligible(entity, ontology, fixed))
        if len({task["task_id"] for task in tasks}) != len(tasks):
            raise ValueError("Duplicate concept-review tasks.")
        packs.append({"fold": fold, "tasks": tasks, "validation_document_ids": folds[fold],
                      "upstream_training_document_ids": sorted(set(documents)-set(folds[fold])),
                      "llm_training_document_ids": [], "demonstration_document_ids": [],
                      "answers_removed_before_prompting": True, "baseline_exact_replay": True,
                      "plan_sha256": digest_file(plan_path)})
    for pack in packs:
        directory = root / f"fold{pack['fold']}"
        write_json(directory / "tasks.json", pack)
        manifest = {key: value for key, value in pack.items() if key != "tasks"}
        manifest.update(task_count=len(pack["tasks"]), tasks_sha256=digest_file(directory / "tasks.json"),
                        acronym_tasks=sum(short_form(task["entity"]["text"]) for task in pack["tasks"]),
                        tasks_with_document_definitions=sum(bool(task["document_definitions"]) for task in pack["tasks"]))
        write_json(directory / "prepared.json", manifest)
        print(json.dumps(manifest), flush=True)


def evaluate_pilot(plan_path, plan):
    documents, baseline, folds = inputs(plan)
    report = ROOT / plan["report_dir"]
    if (report / "summary.json").exists():
        raise FileExistsError("Do not overwrite completed concept-review scores.")
    outputs = {policy: [] for policy in plan["policies"]}
    source, original, fold_rows, inference = [], [], [], []
    for fold in plan["pilot_folds"]:
        directory, _, tasks = load_tasks(plan_path, plan, fold)
        summary = json.loads((directory / "summary.json").read_text())
        if summary["status"] != "completed" or summary["plan_sha256"] != digest_file(plan_path) or summary["decisions_sha256"] != digest_file(directory / "decisions.json"):
            raise ValueError("Inference is incomplete, altered, or belongs to another plan.")
        records = json.loads((directory / "decisions.json").read_text())
        index = {row["task_id"]: row for row in records}
        if len(index) != len(records) or set(index) != {task["task_id"] for task in tasks}:
            raise ValueError("Response coverage differs from the frozen tasks.")
        rejected = {str(d): set() for d in folds[fold]}
        for task in tasks:
            record = index[task["task_id"]]
            if any(record[key] != task[key] for key in plan["identity_fields"]):
                raise ValueError("Response identity differs from the original entity.")
            if record["retained_fragments"] != task["fragments"] or record["response_sha256"] != hashlib.sha256(record["response"].encode()).hexdigest():
                raise ValueError("Source evidence or raw response was altered.")
            parsed = parse_decision(record["response"], task["fragments"])
            if parsed != {key: record[key] for key in ["decision", "evidence_ids"]} or record["parse_error"] is not None:
                raise ValueError("Decision parsing failed exact replay.")
            if baseline[task["pmc_id"]]["entities"][task["entity_index"]] != task["entity"]:
                raise ValueError("A task does not match the exact baseline entity.")
            if parsed["decision"] == "other_or_general":
                rejected[task["pmc_id"]].add(task["entity_index"])
        model = association_model(plan, fold, documents, folds[fold])
        validation = [documents[identifier] for identifier in folds[fold]]
        base = [baseline[identifier] for identifier in folds[fold]]
        replay = [from_scores(blind(d), b["entities"], model.predict_scores(blind(d), b["entities"]), 0.5) for d,b in zip(validation,base,strict=True)]
        if replay != base:
            raise ValueError("Baseline no longer replays exactly.")
        source.extend(validation); original.extend(base)
        row = {"fold": fold, "baseline": evaluate(validation, base), "policies": {}}
        for policy in plan["policies"]:
            predicted, edits = [], []
            for document in validation:
                identifier = str(document["pmc_id"])
                entities = []
                for i, entity in enumerate(baseline[identifier]["entities"]):
                    if i in rejected[identifier] and (policy == "all_reviewed" or short_form(entity["text"])):
                        edits.append({"pmc_id": identifier, "entity_index": i, "entity": entity})
                    else:
                        entities.append(entity)
                predicted.append(from_scores(blind(document), entities, model.predict_scores(blind(document), entities), 0.5))
            outputs[policy].extend(predicted)
            metrics = evaluate(validation, predicted)
            row["policies"][policy] = {"metrics": metrics, "removed_entity_records": len(edits)}
            write_json(report / f"fold{fold}_{policy}_edits.json", edits)
            print(json.dumps({"fold": fold, "policy": policy, "score": metrics["score"], "gain": metrics["score"]-row["baseline"]["score"], "removed": len(edits)}), flush=True)
        fold_rows.append(row); inference.append(summary)
        write_json(report / f"fold{fold}_metrics.json", row)
    base = evaluate(source, original)
    ranked, gates = [], plan["promotion"]
    for policy in plan["policies"]:
        metrics = evaluate(source, outputs[policy])
        deltas = [row["policies"][policy]["metrics"]["score"]-row["baseline"]["score"] for row in fold_rows]
        gain = sum(deltas)/len(deltas)
        checks = {"minimum_mean_gain": gain >= gates["minimum_mean_gain"], "worst_fold": min(deltas) >= -gates["maximum_fold_loss"],
                  "mention_improved": metrics["mention"]["f1"] > base["mention"]["f1"],
                  "component_guard": all(metrics[key]["f1"] >= base[key]["f1"]-gates["maximum_component_loss"] for key in ["mention","document","association_micro","association_macro"]),
                  "format_guard": all(item["format_success_rate"] >= gates["minimum_format_success_rate"] for item in inference)}
        ranked.append({"policy": policy, "metrics": metrics, "fold_deltas": deltas, "mean_gain": gain, "checks": checks,
                       "eligible": all(checks.values()), "errors": error_counts(source, outputs[policy]),
                       "removed_true_mention_units": base["mention"]["tp"]-metrics["mention"]["tp"],
                       "removed_false_mention_units": base["mention"]["fp"]-metrics["mention"]["fp"]})
        write_jsonl(report / f"{policy}_predictions.jsonl", outputs[policy])
    ranked.sort(key=lambda row: (-row["mean_gain"], row["policy"]))
    eligible_rows = [row for row in ranked if row["eligible"]]
    best = eligible_rows[0] if eligible_rows else ranked[0]
    validate_file(report / f"{best['policy']}_predictions.jsonl", source, Ontology(ROOT / plan["ontology_path"]), report / "validation.json")
    result = {"plan_sha256": digest_file(plan_path), "baseline": base, "ranked": ranked,
              "selected": best["policy"] if eligible_rows else None, "inference": inference,
              "best_bootstrap": bootstrap_delta(source, original, outputs[best["policy"]]),
              "scope": "Two repeatedly used development folds, not an independent or official B score. Requires remaining-fold confirmation before B promotion."}
    write_json(report / "plan.json", plan); write_json(report / "summary.json", result)
    print(json.dumps({"selected": result["selected"], "best_mean_gain": best["mean_gain"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare","preflight","infer","evaluate"])
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/llm_concept_review/plan.json")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.stage == "prepare": prepare(args.plan, plan)
    elif args.stage == "evaluate": evaluate_pilot(args.plan, plan)
    else: infer(args.plan, plan, args.fold, preflight=args.stage == "preflight")
