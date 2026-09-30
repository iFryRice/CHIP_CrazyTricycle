"""Compare the predeclared BIO pilot policies using the full PatientPheX score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.association import AssociationModel
from patientphex.data import digest_file, prediction_record, read_jsonl, write_json, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.semantic_linking import SemanticIndex
from patientphex.span_linking import SpanLinker
from scripts.evaluate_semantic_linking import eligible_spans, proposals, verify_upstream_partition
from scripts.optimize_cpu_association import blind, error_counts, validate_file


def merge_entities(base_entities, candidates, confidence, similarity, margin):
    accepted = list(base_entities)
    for candidate in sorted(candidates, key=lambda item: (-item["span_score"] * item["similarity"], item["entity"]["offset"], item["entity"]["length"])):
        if candidate["span_score"] < confidence:
            continue
        if candidate["source"] == "semantic" and (candidate["similarity"] < similarity or candidate["margin"] < margin):
            continue
        entity = candidate["entity"]
        if any(entity["offset"] < old["offset"] + old["length"] and old["offset"] < entity["offset"] + entity["length"] for old in accepted):
            continue
        accepted.append(entity)
    return sorted(accepted, key=lambda e: (e["offset"], e["length"]))


def run(args):
    started = time.time()
    plan = json.loads((ROOT / "experiments/bio_ner/plan.json").read_text())
    evaluation_plan = json.loads(args.plan.read_text())
    for path, expected in {**evaluation_plan["input_sha256"], **evaluation_plan["code_sha256"]}.items():
        if digest_file(ROOT / path) != expected:
            raise ValueError(f"Pilot evaluation input/code mismatch: {path}")
    report = ROOT / evaluation_plan["report_dir"]
    if report.exists():
        raise FileExistsError("Refusing to replace an earlier BIO evaluation.")
    documents = read_jsonl(ROOT / plan["train_path"])
    document_index = {str(d["pmc_id"]): d for d in documents}
    splits = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    pilot_folds = plan["pilot_folds"]
    records_by_fold, run_hashes = {}, {}
    for fold in pilot_folds:
        directory = ROOT / f"work/span_ner/bio-pilot-fold{fold}"
        manifest = json.loads((directory / "manifest.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        if manifest["smoke"] or manifest["fit_all"] or summary["status"] != "completed" or summary["epochs_completed"] != 5:
            raise ValueError("The BIO pilot did not complete the declared protocol.")
        if manifest["plan_sha256"] != digest_file(ROOT / "experiments/bio_ner/plan.json") or summary["plan_sha256"] != manifest["plan_sha256"]:
            raise ValueError("BIO pilot training plan identity changed.")
        verify_upstream_partition(manifest, splits[fold], document_index, set())
        if digest_file(directory / "last.pt") != summary["checkpoint_sha256"]:
            raise ValueError("BIO pilot checkpoint changed after completion.")
        records = read_jsonl(directory / "last_validation_spans.jsonl")
        if {str(d["pmc_id"]) for d in records} != set(splits[fold]):
            raise ValueError("BIO predictions have incorrect validation coverage.")
        records_by_fold[fold] = records
        run_hashes[str(fold)] = {name: digest_file(directory / name) for name in ("manifest.json", "summary.json", "last_validation_spans.jsonl")}
    report.mkdir(parents=True)
    write_json(report / "plan.json", evaluation_plan)
    ontology = Ontology(ROOT / evaluation_plan["ontology_path"])
    model_manifest = json.loads((ROOT / "experiments/semantic_linking/model.json").read_text())
    for filename, expected in model_manifest["files"].items():
        if digest_file(ROOT / "models/sapbert" / filename) != expected["sha256"]:
            raise ValueError("SapBERT weights/tokenizer changed.")
    identity = {"model_manifest_sha256": digest_file(ROOT / "experiments/semantic_linking/model.json"),
                "ontology_sha256": digest_file(ROOT / evaluation_plan["ontology_path"]),
                "umls_sha256": digest_file(ROOT / evaluation_plan["umls_path"]),
                "implementation_sha256": digest_file(ROOT / "patientphex/semantic_linking.py")}
    semantic = SemanticIndex(ROOT / "models/sapbert", ROOT / "work/span_ner/semantic_linking/index", ontology, ROOT / evaluation_plan["umls_path"], identity)
    grid = plan["pilot_inference_grid"]
    minimum = min(grid["bio_confidence"])
    queries = sorted({span["text"] for records in records_by_fold.values() for record in records for span in eligible_spans(record, minimum)})
    retrieval = semantic.retrieve(queries)
    baseline_index = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["baseline_oof"])}
    cpu_index = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / evaluation_plan["cpu_oof"])}
    configurations = [{"name": f"{policy}_t{threshold:g}", "entity_policy": policy, "confidence": threshold}
                      for policy in grid["entity_policy"] for threshold in grid["bio_confidence"]]
    variants = {config["name"]: [] for config in configurations}
    fold_results, pilot_documents = [], []
    for fold in pilot_folds:
        validation = [document_index[identifier] for identifier in splits[fold]]
        pilot_documents.extend(validation)
        training = [d for d in documents if str(d["pmc_id"]) not in set(splits[fold])]
        linker = SpanLinker(ontology, ROOT / evaluation_plan["umls_path"]).fit(training)
        candidates = {}
        for record in records_by_fold[fold]:
            identifier = str(record["pmc_id"])
            proposed = proposals(blind(document_index[identifier]), record, linker, retrieval, minimum)
            for item in proposed:
                resolution = linker.resolve(item["entity"]["text"])
                if resolution["identifier"] == "-1" and resolution["source"].startswith("training") and not resolution.get("ambiguous", False):
                    item["entity"]["identifier"] = "-1"
                    item.update(source="known_unmapped", similarity=1.0, margin=1.0)
            candidates[identifier] = proposed
        fold_dir = report / f"fold{fold}"
        fold_dir.mkdir()
        write_json(fold_dir / "candidates.json", candidates)
        base_metrics = evaluate(validation, [baseline_index[str(d["pmc_id"])] for d in validation])
        row = {"fold": fold, "baseline": base_metrics, "configurations": {}}
        for config in configurations:
            predictions = []
            for document in validation:
                identifier = str(document["pmc_id"])
                initial = [] if config["entity_policy"] == "bio_only" else cpu_index[identifier]["entities"]
                entities = merge_entities(initial, candidates[identifier], config["confidence"], grid["semantic_similarity"], grid["semantic_margin"])
                associations = AssociationModel(mode="nearest").predict(blind(document), entities)
                predictions.append(prediction_record(document, entities, associations))
            metrics = evaluate(validation, predictions)
            row["configurations"][config["name"]] = metrics
            variants[config["name"]].extend(predictions)
            print(f"fold={fold} policy={config['name']} score={metrics['score']:.6f} gain={metrics['score']-base_metrics['score']:.6f}", flush=True)
        write_json(fold_dir / "metrics.json", row)
        fold_results.append(row)
    baseline_mean = sum(row["baseline"]["score"] for row in fold_results) / len(fold_results)
    ranked = []
    for config in configurations:
        deltas = [row["configurations"][config["name"]]["score"] - row["baseline"]["score"] for row in fold_results]
        mean_gain = sum(deltas) / len(deltas)
        gates = plan["pilot_advance_gate"]
        checks = {"mean_gain": mean_gain >= gates["minimum_mean_full_pipeline_gain"], "fold_loss": min(deltas) >= -gates["maximum_single_fold_loss"]}
        ranked.append({"configuration": config, "mean_score": baseline_mean + mean_gain, "mean_gain": mean_gain,
                       "fold_deltas": deltas, "metrics": evaluate(pilot_documents, variants[config["name"]]),
                       "checks": checks, "eligible": all(checks.values()), "errors": error_counts(pilot_documents, variants[config["name"]])})
    ranked.sort(key=lambda row: (-row["mean_score"], row["configuration"]["name"]))
    qualified = [row for row in ranked if row["eligible"]]
    selected = qualified[0] if qualified else None
    best = selected or ranked[0]
    write_jsonl(report / "best_pilot_predictions.jsonl", variants[best["configuration"]["name"]])
    validate_file(report / "best_pilot_predictions.jsonl", pilot_documents, ontology, report / "validation.json")
    summary = {"baseline_mean_score": baseline_mean, "ranked": ranked, "selected": selected["configuration"] if selected else None,
               "pilot_folds": pilot_folds, "training_artifacts": run_hashes, "evaluation_plan_sha256": digest_file(args.plan),
               "elapsed_seconds": time.time() - started,
               "scope": "Two-fold pilot on reused development documents; neither a fivefold score nor official B accuracy. No B prediction generated by this pilot."}
    write_json(report / "summary.json", summary)
    print(json.dumps({"selected": summary["selected"], "best_mean_score": best["mean_score"], "mean_gain": best["mean_gain"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/bio_ner/evaluation_plan.json")
    run(parser.parse_args())
