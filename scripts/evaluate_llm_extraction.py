"""Evaluate fixed LLM additions while replaying the unchanged patient linker."""

import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.patient_linking import _check_provenance
from patientphex.semantic_linking import SemanticIndex, merge_neural_additions
from patientphex.span_linking import SpanLinker
from scripts.evaluate_semantic_linking import proposals, verify_upstream_partition
from scripts.optimize_cpu_association import blind, bootstrap_delta, error_counts, from_scores, validate_file


def run():
    started = time.time()
    plan_path = ROOT / "experiments/llm_extraction/plan.json"
    plan = json.loads(plan_path.read_text())
    for filename, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if digest_file(ROOT / filename) != expected:
            raise ValueError(f"Frozen input/code mismatch: {filename}")
    report = ROOT / plan["report_dir"]
    if report.exists():
        raise FileExistsError("Refusing to overwrite completed or partial evaluation.")
    documents = read_jsonl(ROOT / plan["train_path"])
    lookup = {str(document["pmc_id"]): document for document in documents}
    splits = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    baseline = {str(document["pmc_id"]): document for document in read_jsonl(ROOT / plan["baseline_oof"])}
    ontology = Ontology(ROOT / plan["ontology_path"])
    all_records, inference = {}, []
    for fold in plan["pilot_folds"]:
        directory = ROOT / plan["work_dir"] / f"fold{fold}"
        summary = json.loads((directory / "summary.json").read_text())
        verify_upstream_partition(summary, splits[fold], lookup, set())
        if summary["status"] != "completed" or summary["plan_sha256"] != digest_file(plan_path):
            raise ValueError("Inference did not complete this frozen experiment.")
        if digest_file(directory / "spans.jsonl") != summary["span_sha256"] or digest_file(directory / "prompt_audit.json") != summary["prompt_audit_sha256"]:
            raise ValueError("Saved inference artifacts changed after completion.")
        records = read_jsonl(directory / "spans.jsonl")
        if {str(record["pmc_id"]) for record in records} != set(splits[fold]):
            raise ValueError("Inference coverage differs from the held-out fold.")
        all_records[fold] = records
        inference.append(summary)
    model_manifest = json.loads((ROOT / "experiments/semantic_linking/model.json").read_text())
    for filename, entry in model_manifest["files"].items():
        if digest_file(ROOT / "models/sapbert" / filename) != entry["sha256"]:
            raise ValueError("The fixed SapBERT checkpoint changed.")
    identity = {"model_manifest_sha256": digest_file(ROOT / "experiments/semantic_linking/model.json"),
                "ontology_sha256": digest_file(ROOT / plan["ontology_path"]), "umls_sha256": digest_file(ROOT / plan["umls_path"]),
                "implementation_sha256": digest_file(ROOT / "patientphex/semantic_linking.py")}
    semantic = SemanticIndex(ROOT / "models/sapbert", ROOT / "work/span_ner/semantic_linking/index", ontology, ROOT / plan["umls_path"], identity)
    queries = sorted({span["text"] for records in all_records.values() for record in records for span in record["spans"]})
    retrieved = semantic.retrieve(queries)
    report.mkdir(parents=True)
    write_json(report / "plan.json", plan)
    write_json(report / "retrieved.json", retrieved)
    predictions = {config["name"]: [] for config in plan["configurations"]}
    fold_results, pilot_documents, pilot_baseline = [], [], []
    for fold in plan["pilot_folds"]:
        validation = [lookup[identifier] for identifier in splits[fold]]
        training = [document for document in documents if str(document["pmc_id"]) not in splits[fold]]
        pilot_documents.extend(validation)
        original = [baseline[str(document["pmc_id"])] for document in validation]
        pilot_baseline.extend(original)
        model_path = ROOT / f"work/span_ner/document_abbreviations/fold{fold}.pkl"
        if digest_file(model_path) != plan["association_model_sha256"][str(fold)]:
            raise ValueError("The frozen patient association model changed.")
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        _check_provenance(model.provenance_, set(lookup) - set(splits[fold]), set(splits[fold]))
        reproduced = [from_scores(document, baseline[str(document["pmc_id"])]["entities"],
                       model.predict_scores(blind(document), baseline[str(document["pmc_id"])]["entities"]), 0.5) for document in validation]
        if reproduced != original:
            raise ValueError("The existing baseline no longer reproduces exactly.")
        linker = SpanLinker(ontology, ROOT / plan["umls_path"]).fit(training)
        candidates = {str(record["pmc_id"]): proposals(blind(lookup[str(record["pmc_id"])]), record, linker, retrieved, 1.0)
                      for record in all_records[fold]}
        write_json(report / f"fold{fold}_candidates.json", candidates)
        row = {"fold": fold, "baseline": evaluate(validation, original), "configurations": {}}
        for config in plan["configurations"]:
            local, added = [], 0
            for document in validation:
                identifier = str(document["pmc_id"])
                base_entities = baseline[identifier]["entities"]
                entities = merge_neural_additions(base_entities, candidates[identifier], 1.0, config["semantic_threshold"], config["margin_threshold"])
                added += len(entities) - len(base_entities)
                local.append(from_scores(document, entities, model.predict_scores(blind(document), entities), 0.5))
            predictions[config["name"]].extend(local)
            metrics = evaluate(validation, local)
            row["configurations"][config["name"]] = {"metrics": metrics, "added_mentions": added}
            print(json.dumps({"fold": fold, "configuration": config["name"], "score": metrics["score"],
                              "gain": metrics["score"] - row["baseline"]["score"], "added_mentions": added}), flush=True)
        fold_results.append(row)
        write_json(report / f"fold{fold}_metrics.json", row)
    base_metrics = evaluate(pilot_documents, pilot_baseline)
    ranked = []
    gate = plan["promotion"]
    for config in plan["configurations"]:
        current = predictions[config["name"]]
        metrics = evaluate(pilot_documents, current)
        deltas = [row["configurations"][config["name"]]["metrics"]["score"] - row["baseline"]["score"] for row in fold_results]
        mean_gain = sum(deltas) / len(deltas)
        checks = {"mean_gain": mean_gain >= gate["minimum_mean_gain"], "worst_fold": min(deltas) >= -gate["maximum_fold_loss"],
                  "mention_improved": metrics["mention"]["f1"] > base_metrics["mention"]["f1"],
                  "association_guard": all(metrics[name]["f1"] >= base_metrics[name]["f1"] - gate["maximum_association_loss"] for name in ["association_micro", "association_macro"]),
                  "format_success": all(summary["format_success_rate"] >= gate["minimum_format_success_rate"] for summary in inference)}
        ranked.append({"configuration": config, "metrics": metrics, "mean_gain": mean_gain, "fold_deltas": deltas,
                       "checks": checks, "eligible": all(checks.values()), "errors": error_counts(pilot_documents, current)})
    ranked.sort(key=lambda row: (-row["mean_gain"], row["configuration"]["name"]))
    eligible = [row for row in ranked if row["eligible"]]
    best = eligible[0] if eligible else ranked[0]
    best_predictions = predictions[best["configuration"]["name"]]
    write_jsonl(report / "best_pilot_predictions.jsonl", best_predictions)
    validate_file(report / "best_pilot_predictions.jsonl", pilot_documents, ontology, report / "validation.json")
    result = {"plan_sha256": digest_file(plan_path), "baseline": base_metrics, "ranked": ranked,
              "selected": best["configuration"] if eligible else None, "best_bootstrap": bootstrap_delta(pilot_documents, pilot_baseline, best_predictions),
              "inference": inference, "elapsed_seconds": time.time() - started,
              "scope": "Two repeatedly used development folds only. A passing policy requires remaining-fold confirmation before any B prediction."}
    write_json(report / "summary.json", result)
    print(json.dumps({"selected": result["selected"], "best_mean_gain": best["mean_gain"]}), flush=True)


if __name__ == "__main__":
    run()
