"""Measure confidence-filtered UMLS recall additions against the scored CPU B method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.association import AssociationModel
from patientphex.data import digest_file, document_folds, prediction_record, read_jsonl, write_json, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.umls_augmentation import UmlsAugmentedExtractor, augment_entities, load_unambiguous_aliases
from scripts.optimize_cpu_association import assert_partition, blind, bootstrap_delta, error_counts, validate_file


def predict(document, base_entities, additions, threshold):
    entities = augment_entities(base_entities, additions, threshold)
    return prediction_record(document, entities, AssociationModel(mode="nearest").predict(blind(document), entities))


def run(plan_path):
    started = time.time()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for path, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if digest_file(ROOT / path) != expected:
            raise ValueError(f"Frozen input/code hash mismatch: {path}")
    report, output = ROOT / plan["report_dir"], ROOT / plan["output"]
    if report.exists() or output.exists():
        raise FileExistsError("Refusing to overwrite an earlier experiment.")
    report.mkdir(parents=True)
    write_json(report / "plan.json", plan)
    documents = read_jsonl(ROOT / plan["train_path"])
    targets = read_jsonl(ROOT / plan["target_path"])
    target_ids = {str(d["pmc_id"]) for d in targets}
    if len(targets) != 100 or sum(len(d["patient"]) for d in targets) != 244 or any(d.get("entities") or d.get("association") for d in targets):
        raise ValueError("Expected the complete blind B set.")
    ontology = Ontology(ROOT / plan["ontology_path"])
    aliases = load_unambiguous_aliases(ROOT / plan["umls_path"], ontology)
    folds = document_folds(documents, folds=5, seed=20260927)
    assert_partition(documents, folds, target_ids)
    expected_split = json.loads((ROOT / plan["split_path"]).read_text(encoding="utf-8"))["folds"]
    if [[str(d["pmc_id"]) for d in fold] for fold in folds] != expected_split:
        raise ValueError("Changed outer folds.")
    original = read_jsonl(ROOT / plan["baseline_oof"])
    baseline_index = {d["pmc_id"]: d for d in original}
    baseline = evaluate(documents, original)
    variants = {str(threshold): [] for threshold in plan["thresholds"]}
    fold_results = []
    for index, validation in enumerate(folds):
        val_ids = {d["pmc_id"] for d in validation}
        training = [d for d in documents if d["pmc_id"] not in val_ids]
        model = UmlsAugmentedExtractor(ontology, aliases).fit(training)
        scored = {d["pmc_id"]: model.score_additions(blind(d)) for d in validation}
        fold_dir = report / f"fold{index}"
        fold_dir.mkdir()
        write_json(fold_dir / "scored_additions.json", scored)
        write_json(fold_dir / "fit.json", {"training_document_ids": [str(d["pmc_id"]) for d in training], "validation_document_ids": sorted(str(i) for i in val_ids), "statistics": model.stats, "new_lexical_keys": len(model.added_keys)})
        fold_base = evaluate(validation, [baseline_index[d["pmc_id"]] for d in validation])
        row = {"fold": index, "baseline": fold_base, "configurations": {}}
        for threshold in plan["thresholds"]:
            predictions = [predict(d, baseline_index[d["pmc_id"]]["entities"], scored[d["pmc_id"]], threshold) for d in validation]
            variants[str(threshold)].extend(predictions)
            metrics = evaluate(validation, predictions)
            row["configurations"][str(threshold)] = metrics
            print(f"fold={index} threshold={threshold} score={metrics['score']:.6f} gain={metrics['score']-fold_base['score']:.6f}", flush=True)
        write_json(fold_dir / "metrics.json", row)
        fold_results.append(row)
    ranked = []
    for threshold in plan["thresholds"]:
        predictions = variants[str(threshold)]
        metrics = evaluate(documents, predictions)
        deltas = [row["configurations"][str(threshold)]["score"] - row["baseline"]["score"] for row in fold_results]
        gates = plan["promotion"]
        checks = {
            "minimum_gain": metrics["score"] - baseline["score"] >= gates["minimum_gain"],
            "fold_consistency": sum(delta >= 0 for delta in deltas) >= gates["minimum_nonworse_folds"],
            "mention_improved": metrics["mention"]["f1"] > baseline["mention"]["f1"],
            "document_improved": metrics["document"]["f1"] > baseline["document"]["f1"],
            "association_guard": all(metrics[m]["f1"] >= baseline[m]["f1"] - gates["maximum_association_f1_loss"] for m in ("association_micro", "association_macro")),
        }
        ranked.append({"threshold": threshold, "metrics": metrics, "fold_deltas": deltas, "checks": checks,
                       "eligible": all(checks.values()), "errors": error_counts(documents, predictions)})
    ranked.sort(key=lambda row: (-row["metrics"]["score"], -row["threshold"]))
    eligible = [row for row in ranked if row["eligible"]]
    selected = eligible[0] if eligible else None
    best = selected or ranked[0]
    summary = {"plan_sha256": digest_file(plan_path), "baseline": baseline, "baseline_errors": error_counts(documents, original),
               "ranked": ranked, "selected_threshold": selected["threshold"] if selected else None,
               "unambiguous_aliases": len(aliases), "official_baseline": plan["official_baseline"],
               "score_scope": "Repeated development CV; not independent test or B accuracy."}
    best_predictions = variants[str(best["threshold"])]
    summary["best_bootstrap"] = bootstrap_delta(documents, original, best_predictions)
    write_jsonl(report / "best_development_oof.jsonl", best_predictions)
    validate_file(report / "best_development_oof.jsonl", documents, ontology, report / "oof_validation.json")
    if selected:
        threshold = selected["threshold"]
        model = UmlsAugmentedExtractor(ontology, aliases).fit(documents)
        baseline_b = {d["pmc_id"]: d for d in read_jsonl(ROOT / plan["baseline_b"])}
        predictions = [predict(d, baseline_b[d["pmc_id"]]["entities"], model.score_additions(blind(d)), threshold) for d in targets]
        write_jsonl(output, predictions)
        summary["b_validation"] = validate_file(output, targets, ontology, report / "b_validation.json")
        summary["output_sha256"] = digest_file(output)
        write_json(report / "final_fit.json", {"training_document_ids": [str(d["pmc_id"]) for d in documents], "statistics": model.stats, "new_lexical_keys": len(model.added_keys), "threshold": threshold})
        model_path = ROOT / "work/span_ner/umls_augmentation/entity_filter.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as stream:
            pickle.dump(model, stream)
        with model_path.open("rb") as stream:
            restored = pickle.load(stream)
        for document, expected in zip(targets, predictions, strict=True):
            actual = predict(document, baseline_b[document["pmc_id"]]["entities"], restored.score_additions(blind(document)), threshold)
            if actual != expected:
                raise ValueError("Serialized model failed B prediction replay.")
        summary["model_sha256"] = digest_file(model_path)
        summary["model_replay_verified"] = True
    summary["elapsed_seconds"] = time.time() - started
    write_json(report / "summary.json", summary)
    print(json.dumps({"selected_threshold": summary["selected_threshold"], "best_score": best["metrics"]["score"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/umls_augmentation/plan.json")
    run(parser.parse_args().plan)
