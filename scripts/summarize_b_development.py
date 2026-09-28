"""Audit five saved development folds and compare frozen CPU OOF predictions.

This reads completed artifacts only. It never fits a model, performs target
inference, or selects parameters using B data. Fold 0 selected this pipeline;
overlapping cross-fold training makes the aggregate a development estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patientphex.data import read_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.validation import validate_submission

METRICS = ("mention", "document", "association_micro", "association_macro")
CHOICES = ("entity_strategy", "association_threshold", "span_threshold")
SCOPE = "Five-fold development estimate with local approximate four metrics; fold 0 selected the pipeline and training overlaps across folds. NOT independent test performance or a B leaderboard result."


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_choices(actual: dict, expected: dict, context: str) -> None:
    for key in CHOICES:
        require(key in actual and actual[key] == expected[key], f"{context}: frozen {key} mismatch")


def check_metrics(actual: dict, expected: dict, context: str) -> None:
    for group in METRICS:
        for key, value in actual[group].items():
            if isinstance(value, (int, float)):
                other = expected[group].get(key)
                require(isinstance(other, (int, float)) and math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-12),
                        f"{context}: saved metric mismatch for {group}.{key}")
    require(math.isclose(actual["score"], expected["score"], rel_tol=1e-12, abs_tol=1e-12), f"{context}: saved score mismatch")


def metric_differences(actual: dict, baseline: dict) -> dict:
    return {**{name: actual[name]["f1"] - baseline[name]["f1"] for name in METRICS},
            "score": actual["score"] - baseline["score"]}


def grouped_metrics(documents: list[dict], predictions: list[dict]) -> dict:
    by_id = {str(row["pmc_id"]): row for row in predictions}
    groups = {}
    for name, multiple in (("single_patient", False), ("multiple_patients", True)):
        selected = [doc for doc in documents if (len(doc["patient"]) > 1) == multiple]
        groups[name] = evaluate(selected, [by_id[str(doc["pmc_id"])] for doc in selected]) if selected else None
    return groups


def run(args: argparse.Namespace) -> dict:
    output = args.output_dir.resolve()
    summary_path = output / "development_summary.json"
    oof_path = output / "development_oof.jsonl"
    for path in (summary_path, oof_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite development output: {path}")
    selection_path = args.selection.resolve()
    selection = load_json(selection_path)
    selection_hash = digest(selection_path)
    require(selection.get("phase") == "B", "Frozen pipeline must explicitly declare phase B")
    expected = selection["selected"]
    require(all(key in expected for key in CHOICES), "Frozen pipeline is missing entity/association/span choices")
    require(expected["entity_strategy"] in {"neural_all", "neural_supported", "knowledge_union"}, "Invalid frozen entity strategy")
    for key in ("association_threshold", "span_threshold"):
        require(type(expected[key]) in (int, float) and math.isfinite(expected[key]) and 0 <= expected[key] <= 1,
                f"Invalid frozen {key}")
    documents = read_jsonl(args.data)
    official_ids = {str(doc["pmc_id"]) for doc in documents}
    require(len(documents) == 80 and len(official_ids) == 80, "Development data must contain exactly 80 unique official training documents")
    data_hash, ontology_hash = digest(args.data), digest(args.ontology)
    resources = selection["resource_sha256"]
    require(data_hash == resources["training_data"], "Official training data hash differs from the frozen pipeline")
    require(ontology_hash == resources["ontology"], "Ontology hash differs from the frozen pipeline")
    ontology = Ontology(args.ontology)
    scorer_hash = digest(ROOT / "patientphex/evaluation.py")
    source_by_id = {str(doc["pmc_id"]): doc for doc in documents}
    prediction_map, seen_validation = {}, set()
    folds, input_hashes, optional_artifact_checks = [], {}, []
    association_parameters = selection["association_parameters"]
    require(association_parameters.get("C") == 0.5, "Saved evaluation runner uses C=0.5; frozen configuration differs")

    for fold in range(5):
        name = "linked_boundary-cap100-fold0" if fold == 0 else f"confirmed_boundary-cap100-fold{fold}"
        directory = args.reports_dir / name
        paths = {"config": directory / "config.json", "comparison": directory / "comparison.json",
                 "predictions": directory / "selected_validation_predictions.jsonl"}
        hashes = {key: digest(path) for key, path in paths.items()}
        input_hashes[name] = hashes
        config, comparison = load_json(paths["config"]), load_json(paths["comparison"])
        artifacts = config["artifact_sha256"]
        require(hashes["comparison"] == artifacts.get("comparison.json"), f"Fold {fold}: comparison artifact hash mismatch")
        require(hashes["predictions"] == artifacts.get("selected_validation_predictions.jsonl"), f"Fold {fold}: prediction artifact hash mismatch")
        for filename, expected_hash in artifacts.items():
            require(Path(filename).name == filename and "/" not in filename and "\\" not in filename,
                    f"Fold {fold}: artifact name must be a basename")
            artifact = directory / filename
            if artifact.is_file():
                require(digest(artifact) == expected_hash, f"Fold {fold}: artifact hash mismatch for {filename}")
                optional_artifact_checks.append({"fold": fold, "artifact": filename, "verified": True})
        require(config.get("target_data_read") is False, f"Fold {fold}: target/B exclusion declaration missing or false")
        require(config.get("old_cpu_experiment_rerun") is False, f"Fold {fold}: unexpected old CPU experiment rerun")
        require(config.get("negative_weight") == association_parameters["negative_weight"], f"Fold {fold}: association weight mismatch")
        provenance = config.get("candidate_provenance", {})
        require(provenance.get("kind") == "fixed_knowledge_base" and provenance.get("uses_training_labels") is False
                and provenance.get("training_document_ids") == [], f"Fold {fold}: candidate protocol differs from fixed label-free knowledge")
        for key, frozen_key in (("data", "training_data"), ("ontology", "ontology"), ("umls_terms", "umls_terms"), ("umls_manifest", "umls_manifest")):
            require(config["input_sha256"].get(key) == resources[frozen_key], f"Fold {fold}: {key} resource manifest mismatch")
        require(config["code_sha256"].get("patientphex/evaluation.py") == scorer_hash, f"Fold {fold}: local scorer differs from recorded scorer")
        train_values = [str(value) for value in config["training_document_ids"]]
        validation_values = [str(value) for value in config["validation_document_ids"]]
        train_ids, validation_ids = set(train_values), set(validation_values)
        require(len(train_values) == len(train_ids) == 64 and len(validation_values) == len(validation_ids) == 16,
                f"Fold {fold}: expected 64 unique training and 16 unique validation documents")
        require(not train_ids & validation_ids and train_ids | validation_ids == official_ids,
                f"Fold {fold}: train/validation must partition the official 80 documents")
        require(not validation_ids & seen_validation, f"Fold {fold}: validation documents repeat across folds")
        seen_validation.update(validation_ids)
        check_choices(comparison["selected"], expected, f"Fold {fold}")
        require(comparison.get("approximate_four_metrics") is True and comparison.get("baseline_reproduced") is True,
                f"Fold {fold}: missing completed local-evaluation baseline checks")
        if fold == 0:
            for key, source_key in (("comparison", "comparison"), ("config", "configuration")):
                require(hashes[key] == selection["selection_sources"][source_key]["sha256"], f"Fold 0: frozen selection source {key} hash mismatch")
            require(config["input_sha256"]["manifest"] == selection["selection_sources"]["training_manifest"]["sha256"],
                    "Fold 0: frozen training manifest binding differs")
            require(comparison.get("validation_selection") is True, "Fold 0 must be marked as the parameter-selection fold")
        else:
            frozen = config.get("frozen_selection", {})
            require(frozen.get("sha256") == selection_hash and config["input_sha256"].get("frozen_selection") == selection_hash,
                    f"Fold {fold}: frozen selection file hash mismatch")
            check_choices(frozen.get("selected", {}), expected, f"Fold {fold} config")
            require(frozen.get("current_fold_parameter_selection") is False
                    and comparison.get("current_fold_parameter_selection") is False
                    and comparison.get("frozen_selection") is True and comparison.get("validation_selection") is False,
                    f"Fold {fold}: current fold must not select pipeline parameters")
            require(config["entity_strategies"] == [expected["entity_strategy"]]
                    and config["association_thresholds"] == [expected["association_threshold"]]
                    and len(comparison["comparisons"]) == 1, f"Fold {fold}: expected one frozen configuration")
            check_choices(comparison["comparisons"][0], expected, f"Fold {fold} sole comparison")
        predictions = read_jsonl(paths["predictions"])
        require({str(row["pmc_id"]) for row in predictions} == validation_ids and len(predictions) == 16,
                f"Fold {fold}: predictions do not exactly cover its validation documents")
        gold = [doc for doc in documents if str(doc["pmc_id"]) in validation_ids]
        schema = validate_submission(paths["predictions"], gold, ontology)
        actual = evaluate(gold, predictions)
        check_metrics(actual, comparison["selected"]["metrics"], f"Fold {fold}")
        if fold == 0:
            check_metrics(actual, expected["metrics"], "Frozen fold 0 selection")
        prediction_map.update({str(row["pmc_id"]): row for row in predictions})
        folds.append({"fold": fold, "directory": str(directory.resolve()), "validation_document_ids": validation_values,
                      "training_documents": 64, "validation_documents": 16, "metrics": actual, "schema_valid": schema["valid"],
                      "pipeline_parameter_selection_on_this_fold": fold == 0,
                      "checkpoint_selection_declared": comparison.get("checkpoint_validation_selection"), "artifact_sha256": hashes})

    require(seen_validation == official_ids and set(prediction_map) == official_ids, "Five validation folds do not cover the official 80 documents exactly once")
    ordered_predictions = [prediction_map[str(doc["pmc_id"])] for doc in documents]
    pooled = evaluate(documents, ordered_predictions)
    grouped = grouped_metrics(documents, ordered_predictions)
    cpu_predictions = read_jsonl(args.cpu_oof)
    require(len(cpu_predictions) == 80 and {str(row["pmc_id"]) for row in cpu_predictions} == official_ids,
            "Frozen CPU OOF must cover the same 80 official training documents")
    cpu_validation = validate_submission(args.cpu_oof, documents, ontology)
    cpu_metrics = evaluate(documents, cpu_predictions)
    cpu_groups = grouped_metrics(documents, cpu_predictions)
    scores = [row["metrics"]["score"] for row in folds]
    report = {"status": "complete", "scope": SCOPE, "phase": "B", "target_data_read": False,
              "training_performed": False, "cpu_inference_rerun": False, "official_training_documents": 80,
              "selected_pipeline": {key: expected[key] for key in CHOICES},
              "frozen_selection_path": str(selection_path), "frozen_selection_sha256": selection_hash,
              "pooled_development_metrics": pooled, "patient_count_groups": grouped,
              "fold_score_mean": statistics.mean(scores), "fold_score_std_descriptive": statistics.pstdev(scores), "folds": folds,
              "frozen_cpu_reference": {"path": str(args.cpu_oof.resolve()), "sha256": digest(args.cpu_oof),
                                       "schema_valid": cpu_validation["valid"], "metrics": cpu_metrics, "patient_count_groups": cpu_groups,
                                       "scope": "Existing frozen OOF predictions rescored with the same scorer; no fitting/inference rerun; historic model selection also used these documents"},
              "difference_from_frozen_cpu": metric_differences(pooled, cpu_metrics),
              "group_differences_from_frozen_cpu": {name: metric_differences(grouped[name], cpu_groups[name])
                   if grouped[name] is not None and cpu_groups[name] is not None else None for name in grouped},
              "checks": {"five_validation_folds_disjoint": True, "validation_union_is_exact_official_80": True,
                         "each_training_set_is_validation_complement": True, "all_pipeline_choices_match_frozen_selection": True,
                         "folds_1_to_4_declare_no_parameter_selection": True, "all_folds_declare_no_target_data_read": True,
                         "required_artifact_hashes_verified": True, "shared_scorer_hash_verified": True},
              "provenance": {"training_data_sha256": data_hash, "ontology_sha256": ontology_hash, "scorer_sha256": scorer_hash,
                             "summary_script_sha256": digest(Path(__file__)), "fold_input_sha256": input_hashes,
                             "available_artifact_hash_checks": optional_artifact_checks,
                             "resource_verification_scope": "Training data and ontology files verified locally; UMLS resource hashes checked for consistency against the frozen selection and each fold manifest. Uncopied remote inputs/models are not independently rehashed here."},
              "limitations": ["Fold 0 selected architecture and thresholds; its scores are optimistically selected.",
                              "Training documents overlap across folds, including the selection fold appearing in other folds' training sets; folds are not independent trials.",
                              "The 80 documents have been reused for earlier development; this pooled result is a development estimate, not a held-out test.",
                              "B target exclusion is checked from fold data membership, frozen selection metadata and explicit input declarations; this does not prove arbitrary undisclosed upstream behavior.",
                              "No official scorer is available; retain the approximation notes returned by the local evaluator.",
                              "New pipeline versus frozen CPU differs in multiple components; differences are not single-component causal estimates or statistical significance claims."]}
    # Only the two requested outputs are created; an existing report directory
    # can contain unrelated B artifacts. Exclusive creation prevents overwrites.
    output.mkdir(parents=True, exist_ok=True)
    with oof_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in ordered_predictions:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    report["development_oof_validation"] = validate_submission(oof_path, documents, ontology)
    report["development_oof_sha256"] = digest(oof_path)
    with summary_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"summary": str(summary_path), "oof": str(oof_path), "score": pooled["score"],
            "cpu_score": cpu_metrics["score"], "difference": pooled["score"] - cpu_metrics["score"], "scope": SCOPE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports/span_ner")
    parser.add_argument("--selection", type=Path, default=ROOT / "experiments/span_ner/selected_pipeline.json")
    parser.add_argument("--data", type=Path, default=ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl")
    parser.add_argument("--cpu-oof", type=Path, default=ROOT / "work/gpu_route/cpu_oof_input.jsonl")
    parser.add_argument("--ontology", type=Path, default=ROOT / "PatientPheX-V1-A/hp.obo")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/b_supervised")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
