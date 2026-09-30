"""Evaluate patient-balanced linking on nested CPU dictionary candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.association import AssociationModel, _concepts, _gold_concepts
from patientphex.data import digest_file, document_folds, prediction_record, read_jsonl, write_json, write_jsonl
from patientphex.entities import EntityExtractor
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.patient_linking import PatientLinkingModel, _check_provenance
from patientphex.validation import validate_submission


def blind(document):
    return {**document, "entities": [], "association": []}


def assert_partition(documents, folds, forbidden):
    identifiers = [str(d["pmc_id"]) for d in documents]
    actual = [str(d["pmc_id"]) for fold in folds for d in fold]
    if len(set(identifiers)) != len(identifiers) or sorted(actual) != sorted(identifiers):
        raise ValueError("Fold partition must cover each allowed document exactly once.")
    if set(actual) & set(forbidden):
        raise ValueError("Forbidden documents entered candidate generation.")


def crossfit_candidates(documents, ontology, forbidden, folds=4, seed=20260929):
    partitions = document_folds(documents, folds=folds, seed=seed)
    assert_partition(documents, partitions, forbidden)
    candidates, metadata = {}, {}
    for index, validation in enumerate(partitions):
        held_out = {str(d["pmc_id"]) for d in validation}
        training = [d for d in documents if str(d["pmc_id"]) not in held_out]
        sources = sorted(str(d["pmc_id"]) for d in training)
        extractor = EntityExtractor(ontology, mode="dictionary").fit(training)
        for document in validation:
            identifier = str(document["pmc_id"])
            candidates[identifier] = extractor.predict(blind(document))
            metadata[identifier] = {
                "training_document_ids": sources,
                "label_source_document_ids": sources,
                "inner_fold": index,
                "selection_document_ids": [],
            }
        print(f"inner_fold={index} train={len(training)} held_out={len(validation)}", flush=True)
    provenance = {"kind": "out_of_fold", "by_document": metadata,
                  "method": "Fixed dictionary settings; aliases and statistics fitted inside each inner training split."}
    _check_provenance(provenance, set(candidates), set(forbidden))
    return candidates, provenance


def from_scores(document, entities, scores, threshold):
    associations = [{"patient_id": p["patient_id"], "phenotype": []} for p in document["patient"]]
    targets = {a["patient_id"]: a["phenotype"] for a in associations}
    for item in scores:
        if item["score"] >= threshold:
            targets[item["patient_id"]].append(item["concept"])
    return prediction_record(document, entities, associations)


def error_counts(documents, predictions):
    counts = {"gold": 0, "reachable": 0, "reachable_fn": 0, "unreachable_fn": 0, "fp": 0, "tp": 0}
    index = {d["pmc_id"]: d for d in predictions}
    for document in documents:
        pred = index[document["pmc_id"]]
        concepts = {c for entity in pred["entities"] for c in _concepts(entity)}
        actual = {a["patient_id"]: _gold_concepts(a["phenotype"]) for a in pred["association"]}
        for association in document["association"]:
            gold = _gold_concepts(association["phenotype"])
            proposed = actual[association["patient_id"]]
            counts["gold"] += len(gold)
            counts["reachable"] += len(gold & concepts)
            counts["reachable_fn"] += len((gold - proposed) & concepts)
            counts["unreachable_fn"] += len(gold - concepts)
            counts["fp"] += len(proposed - gold)
            counts["tp"] += len(proposed & gold)
    return counts


def bootstrap_delta(documents, baseline, candidate, seed=20260929, iterations=2000):
    arrays = []
    for predictions in (baseline, candidate):
        index = {d["pmc_id"]: d for d in predictions}
        rows = []
        for document in documents:
            result = evaluate([document], [index[document["pmc_id"]]])
            row = [result[m][k] for m in ("mention", "document", "association_micro") for k in ("tp", "fp", "fn")]
            patients = len(document["patient"])
            rows.append(row + [result["association_macro"]["f1"] * patients, patients])
        arrays.append(np.asarray(rows, dtype=float))
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(documents), size=(iterations, len(documents)))
    scores = []
    for array in arrays:
        totals = array[samples].sum(axis=1)
        f1s = []
        for start in (0, 3, 6):
            denom = 2 * totals[:, start] + totals[:, start + 1] + totals[:, start + 2]
            f1s.append(np.divide(2 * totals[:, start], denom, out=np.zeros_like(denom), where=denom != 0))
        f1s.append(totals[:, 9] / totals[:, 10])
        scores.append(np.mean(f1s, axis=0))
    deltas = scores[1] - scores[0]
    return {"iterations": iterations, "unit": "document", "seed": seed,
            "interval_95": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "fraction_positive": float(np.mean(deltas > 0)),
            "scope": "Descriptive paired bootstrap after model selection, not a selection-adjusted significance test."}


def promotion_checks(base, result, fold_deltas, gates):
    return {
        "minimum_gain": result["score"] - base["score"] >= gates["minimum_gain"],
        "micro_improved": result["association_micro"]["f1"] > base["association_micro"]["f1"],
        "macro_improved": result["association_macro"]["f1"] > base["association_macro"]["f1"],
        "fold_consistency": sum(delta >= 0 for delta in fold_deltas) >= gates["minimum_nonworse_folds"],
        "precision_guard": result["association_micro"]["precision"] >= base["association_micro"]["precision"] - gates["maximum_precision_loss"],
        "entity_metrics_unchanged": all(result[m] == base[m] for m in ("mention", "document")),
    }


def validate_file(path, documents, ontology, report_path):
    result = validate_submission(path, documents, ontology)
    write_json(report_path, result)
    if not result["valid"]:
        raise ValueError(f"Invalid predictions: {result}")
    return result


def run(plan_path):
    started = time.time()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for path, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if digest_file(ROOT / path) != expected:
            raise ValueError(f"Frozen input/code hash mismatch: {path}")
    report = ROOT / plan["report_dir"]
    output = ROOT / plan["output"]
    if report.exists() or output.exists():
        raise FileExistsError("Refusing to overwrite an earlier experiment.")
    report.mkdir(parents=True)
    write_json(report / "plan.json", plan)
    documents = read_jsonl(ROOT / plan["train_path"])
    target = read_jsonl(ROOT / plan["target_path"])
    target_ids = {str(d["pmc_id"]) for d in target}
    if len(target) != 100 or sum(len(d["patient"]) for d in target) != 244:
        raise ValueError("Expected the complete B set.")
    if any(d.get("entities") or d.get("association") for d in target):
        raise ValueError("B targets must be blind.")
    ontology = Ontology(ROOT / plan["ontology_path"])
    partitions = document_folds(documents, folds=5, seed=20260927)
    assert_partition(documents, partitions, target_ids)
    saved_split = json.loads((ROOT / plan["split_path"]).read_text(encoding="utf-8"))
    if [[str(d["pmc_id"]) for d in fold] for fold in partitions] != saved_split["folds"]:
        raise ValueError("Changed outer folds.")
    original = read_jsonl(ROOT / plan["baseline_oof"])
    original_index = {d["pmc_id"]: d for d in original}
    baseline = evaluate(documents, original)
    variants = {c["name"]: [] for c in plan["configurations"]}
    fold_results = []
    final_candidates, final_metadata = {}, {}
    for index, validation in enumerate(partitions):
        val_ids = {str(d["pmc_id"]) for d in validation}
        training = [d for d in documents if str(d["pmc_id"]) not in val_ids]
        print(f"outer_fold={index} nested_generation_start", flush=True)
        inner_candidates, provenance = crossfit_candidates(training, ontology, val_ids | target_ids)
        fold_dir = report / f"fold{index}"
        fold_dir.mkdir()
        write_jsonl(fold_dir / "training_candidates.jsonl", [{"pmc_id": k, "entities": v} for k, v in sorted(inner_candidates.items())])
        write_json(fold_dir / "candidate_provenance.json", provenance)
        extractor = EntityExtractor(ontology, mode="dictionary").fit(training)
        validation_entities = {d["pmc_id"]: extractor.predict(blind(d)) for d in validation}
        train_ids = sorted(str(d["pmc_id"]) for d in training)
        for document in validation:
            entities = validation_entities[document["pmc_id"]]
            reproduced = prediction_record(document, entities, AssociationModel(mode="nearest").predict(blind(document), entities))
            if reproduced != original_index[document["pmc_id"]]:
                raise ValueError("The dictionary+nearest baseline did not reproduce exactly.")
            identifier = str(document["pmc_id"])
            final_candidates[identifier] = entities
            final_metadata[identifier] = {"training_document_ids": train_ids, "label_source_document_ids": train_ids}
        fold_base = evaluate(validation, [original_index[d["pmc_id"]] for d in validation])
        metrics = {}
        for weight in sorted({c["negative_weight"] for c in plan["configurations"]}):
            model = PatientLinkingModel(C=0.5, negative_weight=weight).fit(
                training, inner_candidates, provenance=provenance, forbidden_document_ids=val_ids | target_ids)
            write_json(fold_dir / f"training_weight_{weight:g}.json", model.training_diagnostics_)
            scored = {d["pmc_id"]: model.predict_scores(blind(d), validation_entities[d["pmc_id"]]) for d in validation}
            write_json(fold_dir / f"scores_weight_{weight:g}.json", scored)
            for config in plan["configurations"]:
                if config["negative_weight"] != weight:
                    continue
                predictions = [from_scores(d, validation_entities[d["pmc_id"]], scored[d["pmc_id"]], config["threshold"]) for d in validation]
                variants[config["name"]].extend(predictions)
                metrics[config["name"]] = evaluate(validation, predictions)
                print(f"fold={index} config={config['name']} score={metrics[config['name']]['score']:.6f}", flush=True)
        fold_results.append({"fold": index, "baseline": fold_base, "configurations": metrics})
        write_json(fold_dir / "metrics.json", fold_results[-1])
    ranked = []
    for config in plan["configurations"]:
        predictions = variants[config["name"]]
        result = evaluate(documents, predictions)
        deltas = [fold["configurations"][config["name"]]["score"] - fold["baseline"]["score"] for fold in fold_results]
        gates = promotion_checks(baseline, result, deltas, plan["promotion"])
        ranked.append({"configuration": config, "metrics": result, "fold_deltas": deltas,
                       "checks": gates, "eligible": all(gates.values()), "errors": error_counts(documents, predictions)})
    ranked.sort(key=lambda row: (-row["metrics"]["score"], row["configuration"]["name"]))
    eligible = [row for row in ranked if row["eligible"]]
    summary = {"plan_sha256": digest_file(plan_path), "baseline": baseline,
               "baseline_errors": error_counts(documents, original), "ranked": ranked,
               "selected": eligible[0]["configuration"] if eligible else None,
               "score_scope": "Repeated development CV; not independent test or B accuracy.",
               "official_baseline": plan["official_baseline"], "environment": {"python": platform.python_version(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    best = eligible[0] if eligible else ranked[0]
    best_predictions = variants[best["configuration"]["name"]]
    summary["best_bootstrap"] = bootstrap_delta(documents, original, best_predictions)
    write_jsonl(report / "best_development_oof.jsonl", best_predictions)
    validate_file(report / "best_development_oof.jsonl", documents, ontology, report / "oof_validation.json")
    if eligible:
        selected = eligible[0]["configuration"]
        model = PatientLinkingModel(C=0.5, negative_weight=selected["negative_weight"], threshold=selected["threshold"]).fit(
            documents, final_candidates, provenance={"kind": "out_of_fold", "by_document": final_metadata},
            forbidden_document_ids=target_ids)
        write_json(report / "final_training_diagnostics.json", model.training_diagnostics_)
        write_json(report / "final_candidate_provenance.json", model.provenance_)
        extractor = EntityExtractor(ontology, mode="dictionary").fit(documents)
        old_b = {d["pmc_id"]: d for d in read_jsonl(ROOT / plan["baseline_b"])}
        predictions = []
        for document in target:
            entities = extractor.predict(blind(document))
            if entities != old_b[document["pmc_id"]]["entities"]:
                raise ValueError("B entity extraction changed unexpectedly.")
            predictions.append(prediction_record(document, entities, model.predict(blind(document), entities)))
        write_jsonl(output, predictions)
        summary["b_validation"] = validate_file(output, target, ontology, report / "b_validation.json")
        work = ROOT / "work/span_ner/cpu_crossfit"
        work.mkdir(parents=True, exist_ok=True)
        model_path = work / "patient_linking.pkl"
        with model_path.open("wb") as stream:
            pickle.dump(model, stream)
        with model_path.open("rb") as stream:
            restored = pickle.load(stream)
        if any(restored.predict(blind(d), p["entities"]) != p["association"] for d, p in zip(target, predictions, strict=True)):
            raise ValueError("Serialized model failed prediction replay.")
        summary["model_sha256"] = digest_file(model_path)
        summary["output_sha256"] = digest_file(output)
        summary["model_replay_verified"] = True
    summary["elapsed_seconds"] = time.time() - started
    write_json(report / "summary.json", summary)
    print(json.dumps({"selected": summary["selected"], "best_score": best["metrics"]["score"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/cpu_crossfit/plan.json")
    run(parser.parse_args().plan)
