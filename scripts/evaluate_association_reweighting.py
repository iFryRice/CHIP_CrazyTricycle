"""Run one predeclared association-only comparison with frozen entity outputs."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patientphex.association import _concepts, _gold_concepts
from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.patient_linking import PatientLinkingModel
from patientphex.validation import validate_submission

SCOPE = "Predeclared single-factor comparison on reused development documents; local approximate metrics, not an independent test or B score."


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def checked_file(record: dict) -> Path:
    path = ROOT / record["path"]
    require(digest_file(path) == record["sha256"], f"Input hash mismatch: {path}")
    return path


def check_metrics(actual: dict, expected: dict) -> None:
    require(math.isclose(actual["score"], expected["score"], abs_tol=1e-12), "Baseline score was not reproduced")
    for name in ("mention", "document", "association_micro", "association_macro"):
        for key in ("precision", "recall", "f1"):
            require(math.isclose(actual[name][key], expected[name][key], abs_tol=1e-12),
                    f"Baseline metric differs: {name}.{key}")


def blind(document: dict) -> dict:
    return {**document, "entities": [], "association": []}


def check_partition(training_ids: list[str], validation_ids: list[str], official_ids: set[str]) -> None:
    training, validation = set(training_ids), set(validation_ids)
    require(len(training) == len(training_ids) and len(validation) == len(validation_ids), "Duplicate fold IDs")
    require(not training & validation, "Training and validation IDs overlap")
    require(training | validation == official_ids, "Fold does not partition the official training data")


def replace_associations(documents: list[dict], frozen_predictions: list[dict],
                         model: PatientLinkingModel, threshold: float) -> list[dict]:
    by_id = {str(doc["pmc_id"]): doc for doc in documents}
    require(len(by_id) == len(documents), "Duplicate source documents")
    require(len(frozen_predictions) == len(documents)
            and {str(row["pmc_id"]) for row in frozen_predictions} == set(by_id), "Frozen candidate coverage differs")
    result = []
    for original in frozen_predictions:
        document = by_id[str(original["pmc_id"])]
        require(document["pmid"] == original["pmid"], "Frozen candidate PMID mismatch")
        row = copy.deepcopy(original)
        row["association"] = model.predict(blind(document), row["entities"], threshold=threshold)
        require(row["entities"] == original["entities"], "Association model changed entity candidates")
        result.append(row)
    return result


def relation_audit(documents: list[dict], predictions: list[dict]) -> dict:
    by_id = {str(row["pmc_id"]): row for row in predictions}
    total = reachable = missed = 0
    for doc in documents:
        row = by_id[str(doc["pmc_id"])]
        concepts = {concept for entity in row["entities"] for concept in _concepts(entity)}
        predicted = {item["patient_id"]: _gold_concepts(item["phenotype"]) for item in row["association"]}
        for item in doc["association"]:
            gold = _gold_concepts(item["phenotype"])
            covered = gold & concepts
            total += len(gold)
            reachable += len(covered)
            missed += len(covered - predicted[item["patient_id"]])
    return {"gold_relations": total, "reachable": reachable, "reachable_but_missed": missed}


def grouped_metrics(documents: list[dict], predictions: list[dict]) -> dict:
    by_id = {str(row["pmc_id"]): row for row in predictions}
    result = {}
    for label, multiple in (("single_patient", False), ("multiple_patients", True)):
        group = [doc for doc in documents if (len(doc["patient"]) > 1) == multiple]
        result[label] = evaluate(group, [by_id[str(doc["pmc_id"])] for doc in group]) if group else None
    return result


def promotion_decision(baseline: dict, candidate: dict, fold_deltas: list[float],
                       baseline_audit: dict, candidate_audit: dict, criteria: dict) -> dict:
    checks = {
        "score_gain": candidate["score"] - baseline["score"] >= criteria["minimum_score_gain"],
        "micro_f1_improves": candidate["association_micro"]["f1"] > baseline["association_micro"]["f1"],
        "macro_f1_improves": candidate["association_macro"]["f1"] > baseline["association_macro"]["f1"],
        "fold_consistency": sum(delta >= -1e-12 for delta in fold_deltas) >= criteria["minimum_nonworse_folds"],
        "precision_loss_bounded": (baseline["association_micro"]["precision"] - candidate["association_micro"]["precision"]
                                   <= criteria["maximum_micro_precision_loss"]),
        "reachable_misses_decrease": candidate_audit["reachable_but_missed"] < baseline_audit["reachable_but_missed"],
        "entity_metrics_unchanged": all(candidate[name] == baseline[name] for name in ("mention", "document")),
        "candidate_reachability_unchanged": candidate_audit["reachable"] == baseline_audit["reachable"],
    }
    return {"passed": all(checks.values()), "checks": checks, "criteria": criteria,
            "scope": "Promotion to an additional experimental B candidate, not a proven replacement for the CPU baseline."}


def fit_model(documents: list[dict], candidates: dict, provenance: dict,
              forbidden: set[str], weight: float, plan: dict) -> PatientLinkingModel:
    return PatientLinkingModel(C=plan["fixed"]["C"], negative_weight=weight).fit(
        documents, candidates, provenance=provenance, forbidden_document_ids=forbidden)


def generate_b(plan: dict, output: Path, work: Path, documents: list[dict],
               candidates: dict, provenance: dict, ontology: Ontology) -> dict:
    target = read_jsonl(checked_file(plan["b_target"]))
    baseline_path = checked_file(plan["b_baseline"])
    frozen = read_jsonl(baseline_path)
    require(len(target) == 100 and sum(len(d["patient"]) for d in target) == 244, "Wrong B cohort")
    require(not any(d.get("entities") or d.get("association") for d in target), "B contains answers")
    forbidden = {str(doc["pmc_id"]) for doc in target}
    require(not forbidden & {str(doc["pmc_id"]) for doc in documents}, "B overlaps training")
    validate_submission(baseline_path, target, ontology)
    baseline_model = fit_model(documents, candidates, provenance, forbidden, plan["baseline_negative_weight"], plan)
    reproduced = replace_associations(target, frozen, baseline_model, plan["fixed"]["association_threshold"])
    require(reproduced == frozen, "Full-training baseline does not reproduce the frozen B artifact")
    model = fit_model(documents, candidates, provenance, forbidden, plan["candidate_negative_weight"], plan)
    predictions = replace_associations(target, frozen, model, plan["fixed"]["association_threshold"])
    target_path = ROOT / plan["b_output"]
    require(not target_path.exists(), "B experiment output already exists")
    pending = output / "validated_b_candidate.jsonl"
    write_jsonl(pending, predictions)
    validation = validate_submission(pending, target, ontology)
    require(validation["document_order_preserved"], "B document order changed")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pending.rename(target_path)
    validation["path"] = str(target_path)
    with (work / "final_patient_linking.pkl").open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(output / "b_validation.json", validation)
    write_json(output / "final_training_diagnostics.json", model.training_diagnostics_)
    return {"status": "completed", "submission": validation, "baseline_reproduced": True,
            "entities_identical_to_previous_B": True, "training_documents": len(documents),
            "model_sha256": digest_file(work / "final_patient_linking.pkl"), "gold_metrics_computed": False}


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    for name, expected in plan["code_sha256"].items():
        require(digest_file(ROOT / name) == expected, f"Code changed after experiment declaration: {name}")
    output, work = ROOT / plan["report_dir"], ROOT / plan["work_dir"]
    require(not output.exists() and not work.exists(), "Experiment outputs are immutable; inspect existing run")
    require(plan["baseline_negative_weight"] == 1.0 and plan["candidate_negative_weight"] == 0.5,
            "This runner implements exactly the predeclared 1.0 versus 0.5 experiment")
    documents = read_jsonl(checked_file(plan["training"]))
    require(len(documents) == 80, "Require the official 80-document cohort")
    official = {str(doc["pmc_id"]) for doc in documents}
    ontology = Ontology(checked_file(plan["ontology"]))
    baseline_summary = json.loads(checked_file(plan["baseline_summary"]).read_text(encoding="utf-8"))
    selection = json.loads(checked_file(plan["baseline_selection"]).read_text(encoding="utf-8"))
    require(plan["fixed"]["association_threshold"] == selection["selected"]["association_threshold"], "Threshold drift")
    require(plan["fixed"]["C"] == selection["association_parameters"]["C"], "Regularization drift")
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    merged_candidates, all_baseline, all_candidate, fold_reports = {}, {}, {}, []
    seen_validation = set()
    shared_provenance = None
    for fold in plan["folds"]:
        directory = ROOT / fold["directory"]
        require(digest_file(directory / "config.json") == fold["config_sha256"], "Fold configuration changed")
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        for name in ("training_candidates.jsonl", "selected_validation_predictions.jsonl", "comparison.json"):
            require(digest_file(directory / name) == config["artifact_sha256"][name], f"Saved fold artifact changed: {name}")
        train_ids, validation_ids = config["training_document_ids"], config["validation_document_ids"]
        check_partition(train_ids, validation_ids, official)
        require(len(train_ids) == 64 and len(validation_ids) == 16, "Unexpected outer-fold sizes")
        require(not seen_validation & set(validation_ids), "Validation folds overlap")
        seen_validation.update(validation_ids)
        training = [doc for doc in documents if str(doc["pmc_id"]) in set(train_ids)]
        validation = [doc for doc in documents if str(doc["pmc_id"]) in set(validation_ids)]
        candidate_rows = read_jsonl(directory / "training_candidates.jsonl")
        candidates = {str(row["pmc_id"]): row["entities"] for row in candidate_rows}
        require(set(candidates) == set(train_ids), "Candidate source includes validation or omits training")
        provenance = config["candidate_provenance"]
        require(provenance["kind"] == "fixed_knowledge_base" and provenance["uses_training_labels"] is False
                and not provenance["training_document_ids"], "Candidate source must be label-free")
        require(config["negative_weight"] == plan["baseline_negative_weight"], "Wrong baseline weighting")
        if shared_provenance is not None:
            require(provenance == shared_provenance, "Knowledge resources changed across folds")
        shared_provenance = provenance
        for identifier, entities in candidates.items():
            require(identifier not in merged_candidates or merged_candidates[identifier] == entities,
                    "Label-free candidates differ between folds")
            merged_candidates[identifier] = entities
        frozen = read_jsonl(directory / "selected_validation_predictions.jsonl")
        validate_submission(directory / "selected_validation_predictions.jsonl", validation, ontology)
        baseline_model = fit_model(training, candidates, provenance, set(validation_ids), plan["baseline_negative_weight"], plan)
        reproduced = replace_associations(validation, frozen, baseline_model, plan["fixed"]["association_threshold"])
        require(reproduced == frozen, "Refitted baseline differs from the saved predictions")
        baseline_metrics = evaluate(validation, reproduced)
        comparison = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
        check_metrics(baseline_metrics, comparison["selected"]["metrics"])
        model = fit_model(training, candidates, provenance, set(validation_ids), plan["candidate_negative_weight"], plan)
        predicted = replace_associations(validation, frozen, model, plan["fixed"]["association_threshold"])
        metrics = evaluate(validation, predicted)
        fold_dir = output / f"fold{fold['fold']}"
        fold_dir.mkdir()
        write_jsonl(fold_dir / "predictions.jsonl", predicted)
        schema = validate_submission(fold_dir / "predictions.jsonl", validation, ontology)
        write_json(fold_dir / "validation.json", schema)
        write_json(fold_dir / "training_diagnostics.json", model.training_diagnostics_)
        with (work / f"fold{fold['fold']}_patient_linking.pkl").open("wb") as stream:
            pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        record = {"fold": fold["fold"], "baseline_reproduced": True, "baseline": baseline_metrics,
                  "candidate": metrics, "score_delta": metrics["score"] - baseline_metrics["score"],
                  "baseline_relation_audit": relation_audit(validation, reproduced),
                  "candidate_relation_audit": relation_audit(validation, predicted),
                  "training_document_ids": train_ids, "validation_document_ids": validation_ids,
                  "frozen_source_config_sha256": fold["config_sha256"], "entities_unchanged": True,
                  "predictions_sha256": schema["sha256"]}
        write_json(fold_dir / "comparison.json", record)
        fold_reports.append(record)
        all_baseline.update({str(doc["pmc_id"]): doc for doc in reproduced})
        all_candidate.update({str(doc["pmc_id"]): doc for doc in predicted})
        print(json.dumps({"fold": fold["fold"], "baseline": baseline_metrics["score"],
                          "candidate": metrics["score"], "delta": record["score_delta"]}), flush=True)
    require(seen_validation == official and set(merged_candidates) == official, "Incomplete five-fold cohort")
    baseline = [all_baseline[str(doc["pmc_id"])] for doc in documents]
    candidate = [all_candidate[str(doc["pmc_id"])] for doc in documents]
    base_metrics, metrics = evaluate(documents, baseline), evaluate(documents, candidate)
    check_metrics(base_metrics, baseline_summary["pooled_development_metrics"])
    base_audit, audit = relation_audit(documents, baseline), relation_audit(documents, candidate)
    gate = promotion_decision(base_metrics, metrics, [row["score_delta"] for row in fold_reports],
                              base_audit, audit, plan["promotion_criteria"])
    write_jsonl(output / "development_oof.jsonl", candidate)
    schema = validate_submission(output / "development_oof.jsonl", documents, ontology)
    result = {"status": "development_completed", "scope": SCOPE, "plan_sha256": digest_file(args.plan),
              "baseline": base_metrics, "candidate": metrics, "score_delta": metrics["score"] - base_metrics["score"],
              "baseline_relation_audit": base_audit, "candidate_relation_audit": audit,
              "folds": fold_reports, "groups": grouped_metrics(documents, candidate),
              "baseline_groups": grouped_metrics(documents, baseline),
              "frozen_cpu_reference": baseline_summary["frozen_cpu_reference"]["metrics"],
              "difference_from_frozen_cpu": metrics["score"] - baseline_summary["frozen_cpu_reference"]["metrics"]["score"],
              "promotion": gate, "development_validation": schema, "entity_predictions_changed": False,
              "threshold_search_performed": False, "target_read_for_selection": False,
              "candidate_protocol": "Existing fixed-KB candidates retained; no nested OOF candidate training in this experiment.",
              "limitations": ["The negative-weight mechanism is tested; candidate-distribution mismatch is not tested.",
                              "All folds and the prior error analysis reuse the same 80 development documents.",
                              "The inherited entity pipeline includes earlier model-selection choices; this is not an independent test."]}
    write_json(output / "comparison.json", result)
    if gate["passed"]:
        result["b_result"] = generate_b(plan, output, work, documents, merged_candidates, shared_provenance, ontology)
    else:
        result["b_result"] = {"status": "not_generated", "reason": "Predeclared promotion gate failed; previous B artifact retained"}
    result.update(status="completed", duration_seconds=time.time() - started)
    write_json(output / "comparison.json", result)
    print(json.dumps({"status": result["status"], "score": metrics["score"], "delta": result["score_delta"],
                      "promotion": gate, "b_result": result["b_result"], "duration_seconds": result["duration_seconds"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/association_reweighting/plan.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
