"""Confirm the frozen acronym-support rule on the remaining BIO folds."""

import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.bio_corroboration import revise_entities
from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.patient_linking import _check_provenance
from scripts.evaluate_semantic_linking import verify_upstream_partition
from scripts.optimize_cpu_association import blind, bootstrap_delta, error_counts, from_scores, validate_file


def run():
    started = time.time()
    plan_path = ROOT / "experiments/bio_corroboration/confirmation_plan.json"
    manifest_path = ROOT / "experiments/bio_corroboration/evaluation_manifest.json"
    specification = json.loads(manifest_path.read_text())
    for filename, expected in {**specification["input_sha256"], **specification["code_sha256"]}.items():
        if digest_file(ROOT / filename) != expected:
            raise ValueError(f"Frozen confirmation input/code mismatch: {filename}")
    plan = json.loads(plan_path.read_text())
    if plan["selected"] != {"name": "veto_acronyms", "veto_acronyms": True, "add_mentions": False}:
        raise ValueError("The selected pilot policy changed.")
    if digest_file(ROOT / plan["training_plan"]) != plan["training_plan_sha256"]:
        raise ValueError("BIO training parameters changed after pilot selection.")
    report = ROOT / "reports/bio_corroboration/confirmation"
    if report.exists():
        raise FileExistsError("Refusing to overwrite confirmation results.")
    documents = read_jsonl(ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl")
    index = {str(d["pmc_id"]): d for d in documents}
    splits = json.loads((ROOT / "reports/cpu_baseline/split.json").read_text())["folds"]
    baseline = read_jsonl(ROOT / plan["baseline_oof"])
    baseline_index = {str(d["pmc_id"]): d for d in baseline}
    baseline_metrics = evaluate(documents, baseline)
    pilot = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / "reports/bio_corroboration/best_pilot_predictions.jsonl")}
    report.mkdir(parents=True)
    predictions, rows, artifacts = [], [], {}
    for fold, val_ids in enumerate(splits):
        validation = [index[identifier] for identifier in val_ids]
        directory = ROOT / plan["runs"][str(fold)]
        manifest = json.loads((directory / "manifest.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        verify_upstream_partition(manifest, val_ids, index, set())
        if manifest["smoke"] or manifest["fit_all"] or manifest["plan_sha256"] != plan["training_plan_sha256"]:
            raise ValueError("BIO run differs from the confirmed training protocol.")
        if summary["status"] != "completed" or summary["epochs_completed"] != 5 or summary["plan_sha256"] != manifest["plan_sha256"]:
            raise ValueError("A BIO fold did not complete the fixed budget.")
        if digest_file(directory / "last.pt") != summary["checkpoint_sha256"]:
            raise ValueError("BIO weights changed after completion.")
        raw = {str(record["pmc_id"]): record["spans"] for record in read_jsonl(directory / "last_validation_spans.jsonl")}
        if set(raw) != set(val_ids):
            raise ValueError("BIO validation coverage differs from the outer fold.")
        model_path = ROOT / f"work/span_ner/document_abbreviations/fold{fold}.pkl"
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        _check_provenance(model.provenance_, set(index) - set(val_ids), set(val_ids))
        reproduced = [from_scores(d, baseline_index[str(d["pmc_id"])]["entities"],
                        model.predict_scores(blind(d), baseline_index[str(d["pmc_id"])]["entities"]), 0.5) for d in validation]
        if reproduced != [baseline_index[str(d["pmc_id"])] for d in validation]:
            raise ValueError("The baseline changed before confirmation.")
        local, edits = [], {}
        for document in validation:
            identifier = str(document["pmc_id"])
            entities, edits[identifier] = revise_entities(baseline_index[identifier]["entities"], raw[identifier], [], veto_acronyms=True)
            prediction = from_scores(document, entities, model.predict_scores(blind(document), entities), 0.5)
            if identifier in pilot and prediction != pilot[identifier]:
                raise ValueError("Frozen policy no longer reproduces the two-fold pilot.")
            local.append(prediction)
        predictions.extend(local)
        base = evaluate(validation, reproduced)
        metrics = evaluate(validation, local)
        row = {"fold": fold, "baseline": base, "metrics": metrics, "delta": metrics["score"] - base["score"]}
        rows.append(row)
        write_json(report / f"fold{fold}_edits.json", edits)
        write_json(report / f"fold{fold}_metrics.json", row)
        artifacts[str(fold)] = {name: digest_file(directory / name) for name in ("manifest.json", "summary.json", "last_validation_spans.jsonl")}
        artifacts[str(fold)]["association_model_sha256"] = digest_file(model_path)
        print(f"fold={fold} score={metrics['score']:.6f} gain={row['delta']:.6f}", flush=True)
    metrics = evaluate(documents, predictions)
    deltas = [rows[fold]["delta"] for fold in plan["confirmation_folds"]]
    gates = plan["promotion"]
    checks = {"confirmation_mean_gain": sum(deltas) / len(deltas) >= gates["minimum_confirmation_mean_gain"],
              "confirmation_consistency": sum(delta >= 0 for delta in deltas) >= gates["minimum_confirmation_nonworse_folds"],
              "confirmation_worst_fold": min(deltas) >= -gates["maximum_confirmation_fold_loss"],
              "full_score_gain": metrics["score"] - baseline_metrics["score"] >= gates["minimum_full_score_gain"],
              "all_components_nonworse": all(metrics[key]["f1"] >= baseline_metrics[key]["f1"] for key in ("mention", "document", "association_micro", "association_macro"))}
    write_jsonl(report / "development_oof.jsonl", predictions)
    validate_file(report / "development_oof.jsonl", documents, Ontology(ROOT / "PatientPheX-V1-A/hp.obo"), report / "validation.json")
    result = {"confirmation_plan_sha256": digest_file(plan_path), "evaluation_manifest_sha256": digest_file(manifest_path),
              "selected": plan["selected"], "baseline": baseline_metrics, "metrics": metrics, "fold_deltas": [row["delta"] for row in rows],
              "confirmation_mean_gain": sum(deltas) / len(deltas), "checks": checks, "promoted": all(checks.values()),
              "errors": error_counts(documents, predictions), "bootstrap": bootstrap_delta(documents, baseline, predictions),
              "artifacts": artifacts, "elapsed_seconds": time.time() - started,
              "scope": "Fivefold development with the rule frozen after two pilot folds. Earlier development reused these documents; this is not an independent test or official B score."}
    write_json(report / "summary.json", result)
    print(json.dumps({"promoted": result["promoted"], "score": metrics["score"], "checks": checks}), flush=True)


if __name__ == "__main__":
    run()
