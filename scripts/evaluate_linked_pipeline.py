"""Compare nine predeclared linked pipelines on a saved document-held-out fold.

This fits a new association model on fixed knowledge-base candidates. It does
not run NER inference, rerun the old CPU pipeline, or read a target/B dataset.
The default mode selects on validation; --frozen-selection evaluates one fixed
development-fold-0 configuration. Neither mode is an independent test estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patientphex.data import prediction_record, read_jsonl, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.patient_linking import PatientLinkingModel
from patientphex.span_linking import SpanLinker
from patientphex.validation import validate_submission
from scripts.analyze_span_thresholds import check_metric_equal, exact_best_threshold, filter_scores
from scripts.train_span_ner import gold_span_sets, score_predictions

LABELS = ("positive", "NO")
STRATEGIES = ("neural_all", "neural_supported", "knowledge_union")
ASSOCIATION_THRESHOLDS = (0.3, 0.5, 0.7)
SCOPE = "Approximate local four-metric evaluation; validation-selected checkpoint, span threshold, entity strategy and association threshold; NOT independent test performance."
FROZEN_SCOPE = "Approximate local four-metric evaluation; span threshold, entity strategy and association threshold frozen from development fold 0, with no parameter selection on this fold. The checkpoint and complete pipeline still involve development selection; NOT independent test performance."


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def blind(document: dict) -> dict:
    return {**document, "entities": [], "association": []}


def best_epoch_metrics(summary: dict, run_dir: Path) -> tuple[dict, str]:
    if summary.get("status") != "completed" or summary.get("smoke") is not False:
        raise ValueError("Require a completed non-smoke training run.")
    if summary["best_epoch"] == summary["epochs_completed"]:
        expected = summary["last_metrics"]
        source = "summary.last_metrics"
    else:
        matches = [row for row in read_jsonl_events(run_dir / "training.jsonl")
                   if row.get("event") == "validation" and row.get("epoch") == summary["best_epoch"]]
        if not matches:
            raise ValueError("Best-epoch validation metrics are missing from training.jsonl.")
        expected = matches[0]["metrics"]
        for row in matches[1:]:
            check_metric_equal(row["metrics"], expected)
        source = f"training.jsonl validation epoch {summary['best_epoch']}"
    if not math.isclose(expected["micro"]["f1"], summary["best_micro_f1"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Best epoch metrics disagree with summary.best_micro_f1.")
    return expected, source


def read_jsonl_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_saved_scores(records: list[dict], documents: list[dict], minimum: float) -> dict:
    sources = {str(document["pmc_id"]): document for document in documents}
    identifiers = [str(record["pmc_id"]) for record in records]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(sources):
        raise ValueError("Saved predictions must exactly cover manifest validation documents.")
    scores = {}
    for record in records:
        identifier = str(record["pmc_id"])
        seen = set()
        for span in record["spans"]:
            offset, length, text = span["offset"], span["length"], span["text"]
            if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0 or length != len(text):
                raise ValueError("Invalid saved span extent.")
            if (offset, length) in seen:
                raise ValueError("Duplicate saved span.")
            seen.add((offset, length))
            if not any(p["offset"] <= offset and offset + length <= p["offset"] + len(p["text"])
                       and p["text"][offset - p["offset"]:offset - p["offset"] + length] == text
                       for p in sources[identifier]["full_text"]):
                raise ValueError("Saved span does not match unchanged original text.")
            labels = span["labels"]
            if not labels or len(set(labels)) != len(labels) or set(labels) != set(span["scores"]) or not set(labels) <= set(LABELS):
                raise ValueError("Invalid saved span labels/scores.")
            for label in labels:
                score = span["scores"][label]
                if type(score) not in (int, float) or not math.isfinite(score) or not minimum <= score <= 1:
                    raise ValueError("Saved score is outside its declared retained range.")
                scores[(identifier, offset, length, LABELS.index(label))] = score
    return scores


def evaluation_plan(saved_scores: dict, validation: list[dict], floor: float,
                    frozen: dict | None = None) -> tuple[dict, tuple[str, ...], tuple[float, ...]]:
    """Choose the development grid, or evaluate exactly one frozen configuration."""
    if frozen is None:
        common = exact_best_threshold(saved_scores, gold_span_sets(validation), floor)
        strategies, association_thresholds = STRATEGIES, ASSOCIATION_THRESHOLDS
    else:
        if floor != 0.5:
            raise ValueError("Frozen evaluation requires the original checkpoint score floor of 0.5.")
        selected = frozen.get("selected") if isinstance(frozen, dict) else None
        if not isinstance(selected, dict):
            raise ValueError("Frozen selection JSON must contain a selected object.")
        strategy = selected.get("entity_strategy")
        association_threshold = selected.get("association_threshold")
        threshold = selected.get("span_threshold")
        if strategy not in STRATEGIES:
            raise ValueError("Frozen selection specifies an unknown entity strategy.")
        for name, value, minimum in (("association_threshold", association_threshold, 0.0),
                                     ("span_threshold", threshold, floor)):
            if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= 1.0:
                raise ValueError(f"Frozen {name} must be finite and within [{minimum}, 1].")
        strategies, association_thresholds = (strategy,), (float(association_threshold),)
        common = {"threshold": float(threshold), "selection": "Frozen from development fold 0; no current-fold threshold search"}
    threshold = common["threshold"]
    if threshold < floor:
        raise ValueError("Selected span threshold is below the saved score floor.")
    common["metrics"] = score_predictions(filter_scores(saved_scores, (threshold, threshold)), validation)
    return common, strategies, association_thresholds


def supported_entities(entities: list[dict], training_unmapped_texts: set[str]) -> list[dict]:
    return [entity for entity in entities if any(part.startswith("HP:") for part in entity["identifier"].split(";"))
            or entity["text"] in training_unmapped_texts]


def knowledge_union(neural_supported: list[dict], knowledge: list[dict], neural_all: list[dict]) -> list[dict]:
    """Keep NER labels at matching extents and deduplicate compound HPO units."""
    neural_notes = {(entity["offset"], entity["length"]): entity["note"] for entity in neural_all}
    result, seen = [], set()
    for original in [*neural_supported, *knowledge]:
        entity = dict(original)
        extent = entity["offset"], entity["length"]
        remaining = []
        for identifier in entity["identifier"].split(";"):
            unit = (*extent, identifier)
            if unit not in seen:
                seen.add(unit)
                remaining.append(identifier)
        if remaining:
            entity["identifier"] = ";".join(remaining)
            entity["note"] = neural_notes.get(extent, entity["note"])
            result.append(entity)
    return sorted(result, key=lambda entity: (entity["offset"], entity["length"], entity["identifier"]))


def associations(document: dict, scores: list[dict], threshold: float) -> list[dict]:
    result = [{"patient_id": patient["patient_id"], "phenotype": []} for patient in document["patient"]]
    targets = {row["patient_id"]: row["phenotype"] for row in result}
    for row in scores:
        if row["score"] >= threshold:
            targets[row["patient_id"]].append(row["concept"])
    return result


def fit_patient_model(training_documents: list[dict], fixed_linker: SpanLinker, *,
                      forbidden_document_ids: set[str], negative_weight: float = 1.0) -> tuple:
    """Use the same label-free candidate protocol in validation and final fitting."""
    if fixed_linker.training_document_ids:
        raise ValueError("Fixed candidate linker must be fitted with no training documents.")
    candidates = {}
    for document in training_documents:
        source = blind(document)
        entities, _ = fixed_linker.link_document(source, fixed_linker.knowledge_spans(source))
        candidates[str(document["pmc_id"])] = entities
    provenance = {"kind": "fixed_knowledge_base", "uses_training_labels": False,
                  "training_document_ids": [], "resource_audit": fixed_linker.resource_audit}
    model = PatientLinkingModel(negative_weight=negative_weight).fit(
        training_documents, candidates, provenance=provenance, forbidden_document_ids=forbidden_document_ids,
    )
    return model, candidates, provenance


def build_entity_variants(document: dict, span_record: dict, fixed_linker: SpanLinker,
                          trained_linker: SpanLinker, threshold: float, trained_unmapped_texts: set[str]) -> tuple:
    """Return identical entity strategy definitions for validation and targets."""
    if fixed_linker.training_document_ids:
        raise ValueError("Knowledge union requires a linker without training aliases.")
    source = blind(document)
    neural, audit = trained_linker.link_document(source, span_record, thresholds={label: threshold for label in LABELS})
    supported = supported_entities(neural, trained_unmapped_texts)
    knowledge, _ = fixed_linker.link_document(source, fixed_linker.knowledge_spans(source))
    knowledge = [entity for entity in knowledge if any(part.startswith("HP:") for part in entity["identifier"].split(";"))]
    return {"neural_all": neural, "neural_supported": supported,
            "knowledge_union": knowledge_union(supported, knowledge, neural)}, audit


def run(args: argparse.Namespace) -> dict:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Immutable output directory already exists: {output}")
    paths = {"data": args.data.resolve(), "ontology": args.ontology.resolve(),
             "umls_terms": args.umls_terms.resolve(),
             "umls_manifest": args.umls_terms.with_name(args.umls_terms.name + ".manifest.json").resolve(),
             "manifest": args.run_dir.resolve() / "manifest.json", "summary": args.run_dir.resolve() / "summary.json",
             "saved_predictions": args.run_dir.resolve() / "best_validation_predictions.jsonl"}
    frozen_path = getattr(args, "frozen_selection", None)
    if frozen_path is not None:
        paths["frozen_selection"] = frozen_path.resolve()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary["best_epoch"] != summary["epochs_completed"]:
        paths["training_log"] = args.run_dir.resolve() / "training.jsonl"
    input_hashes = {name: digest(path) for name, path in paths.items()}
    if input_hashes["data"] != manifest["sha256"]["data"]:
        raise ValueError("Training data hash differs from the saved run manifest.")
    documents = read_jsonl(paths["data"])
    train_list = [str(identifier) for identifier in manifest["training_document_ids"]]
    validation_list = [str(identifier) for identifier in manifest["validation_document_ids"]]
    train_ids, validation_ids = set(train_list), set(validation_list)
    if (len(train_ids) != len(train_list) or len(validation_ids) != len(validation_list)
            or train_ids & validation_ids or train_ids | validation_ids != {str(doc["pmc_id"]) for doc in documents}
            or not train_ids or not validation_ids):
        raise ValueError("Manifest IDs must form a nonempty disjoint complete partition of the training data.")
    training = [doc for doc in documents if str(doc["pmc_id"]) in train_ids]
    validation = [doc for doc in documents if str(doc["pmc_id"]) in validation_ids]
    floor = float(manifest["parameters"]["threshold"])
    if not math.isfinite(floor) or not 0 <= floor <= 1 or summary["threshold"] != floor:
        raise ValueError("Invalid or inconsistent saved score floor.")
    records = read_jsonl(paths["saved_predictions"])
    saved_scores = load_saved_scores(records, validation, floor)
    expected, baseline_source = best_epoch_metrics(summary, args.run_dir)
    baseline = score_predictions(saved_scores, validation)
    check_metric_equal(baseline, expected)
    frozen = json.loads(paths["frozen_selection"].read_text(encoding="utf-8")) if frozen_path is not None else None
    if frozen_path is not None and not isinstance(frozen, dict):
        raise ValueError("Frozen selection JSON must be an object.")
    common, strategies, association_thresholds = evaluation_plan(saved_scores, validation, floor, frozen)
    threshold = common["threshold"]
    scope = FROZEN_SCOPE if frozen_path is not None else SCOPE

    ontology = Ontology(paths["ontology"])
    fixed_linker = SpanLinker(ontology, paths["umls_terms"]).fit([])
    trained_linker = SpanLinker(ontology, paths["umls_terms"]).fit(training, forbidden_document_ids=validation_ids)
    relative_code = ["scripts/evaluate_linked_pipeline.py", "scripts/analyze_span_thresholds.py", "scripts/train_span_ner.py",
                     "patientphex/span_linking.py", "patientphex/patient_linking.py", "patientphex/association.py",
                     "patientphex/entities.py", "patientphex/ontology.py", "patientphex/evaluation.py",
                     "patientphex/validation.py", "patientphex/data.py"]
    code_hashes = {relative: digest(ROOT / relative) for relative in relative_code}
    output.mkdir(parents=True, exist_ok=False)
    print(f"Baseline reproduced; train={len(training)}, validation={len(validation)}, span threshold={threshold:.9g}", flush=True)

    patient_model, training_candidates, provenance = fit_patient_model(
        training, fixed_linker, forbidden_document_ids=validation_ids, negative_weight=args.negative_weight,
    )
    write_jsonl(output / "training_candidates.jsonl", [
        {"pmc_id": doc["pmc_id"], "entities": training_candidates[str(doc["pmc_id"])]} for doc in training
    ])
    write_json(output / "training_diagnostics.json", patient_model.training_diagnostics_)
    with (output / "patient_linking.pkl").open("wb") as stream:
        pickle.dump(patient_model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Association model fitted on {patient_model.training_diagnostics_['candidate_pairs']} fixed-KB pairs", flush=True)

    unmapped_texts = {entity["text"] for doc in training for entity in doc["entities"]
                      if "-1" in entity["identifier"].split(";")}
    records_by_id = {str(record["pmc_id"]): record for record in records}
    entity_sets = {name: {} for name in strategies}
    linking_audit = []
    for document in validation:
        identifier = str(document["pmc_id"])
        variants, audit = build_entity_variants(document, records_by_id[identifier], fixed_linker, trained_linker, threshold, unmapped_texts)
        for strategy in strategies:
            entity_sets[strategy][identifier] = variants[strategy]
        linking_audit.append(audit)
    write_json(output / "linking_diagnostics.json", {"resource_audit": trained_linker.resource_audit,
               "training_alias_audit": trained_linker.fit_audit, "validation": linking_audit})

    comparisons, predictions_by_name = [], {}
    for strategy in strategies:
        score_cache = {str(doc["pmc_id"]): patient_model.predict_scores(blind(doc), entity_sets[strategy][str(doc["pmc_id"])])
                       for doc in validation}
        for association_threshold in association_thresholds:
            suffix = str(association_threshold) if frozen_path is not None else f"{association_threshold:.1f}"
            name = f"{strategy}__association_{suffix}"
            predictions = [prediction_record(doc, entity_sets[strategy][str(doc["pmc_id"])],
                           associations(doc, score_cache[str(doc["pmc_id"])], association_threshold)) for doc in validation]
            metrics = evaluate(validation, predictions)
            comparisons.append({"name": name, "entity_strategy": strategy, "association_threshold": association_threshold,
                                "span_threshold": threshold, "metrics": metrics,
                                "entities": sum(len(doc["entities"]) for doc in predictions),
                                "association_pairs": sum(len(row["phenotype"]) for doc in predictions for row in doc["association"])})
            predictions_by_name[name] = predictions
            print(f"{name}: approximate validation score={metrics['score']:.6f}", flush=True)
    if frozen_path is not None:
        if len(comparisons) != 1:
            raise AssertionError("Frozen evaluation must produce exactly one comparison.")
        selected = comparisons[0]
    else:
        selected = max(comparisons, key=lambda row: (row["metrics"]["score"], row["association_threshold"], -STRATEGIES.index(row["entity_strategy"])))
    selected_path = output / "selected_validation_predictions.jsonl"
    write_jsonl(selected_path, predictions_by_name[selected["name"]])
    validation_report = validate_submission(selected_path, validation, ontology)
    write_json(output / "validation_report.json", validation_report)
    comparison = {"scope": scope, "approximate_four_metrics": True, "validation_selection": frozen_path is None,
                  "baseline_reproduced": True, "baseline_metric_source": baseline_source, "baseline_span_metrics": baseline,
                  "best_epoch": summary["best_epoch"], "saved_score_floor": floor, "selected_common_span_threshold": common,
                  "span_threshold_objective": "Maximize positive/NO exact-span micro F1 on saved validation scores; higher threshold wins ties",
                  "span_label_arbitration": "Linked entities retain one label per extent; highest passing raw score wins, positive on ties",
                  "selection_rule": "Highest approximate four-metric score; ties prefer higher association threshold, then declared entity strategy order",
                  "comparisons": comparisons, "selected": selected}
    if frozen_path is not None:
        comparison.update({"frozen_selection": True, "current_fold_parameter_selection": False,
                           "checkpoint_validation_selection": True,
                           "span_threshold_objective": "No optimization on this fold; evaluate the frozen development-fold-0 span threshold",
                           "selection_rule": "No optimization on this fold; evaluate the single frozen entity strategy and association threshold"})
    write_json(output / "comparison.json", comparison)
    config = {"scope": scope, "paths": {name: str(path) for name, path in paths.items()}, "output_dir": str(output),
              "training_document_ids": train_list, "validation_document_ids": validation_list,
              "entity_strategies": list(strategies), "association_thresholds": list(association_thresholds),
              "negative_weight": args.negative_weight, "candidate_provenance": provenance,
              "neural_supported_unmapped_rule": "Exact original text occurs in a training entity explicitly containing identifier -1",
              "knowledge_union_policy": "Fixed-KB mapped entities plus neural_supported; NER note wins at matching extents; each offset/length/HPO unit appears once",
              "input_sha256": input_hashes, "code_sha256": code_hashes, "original_training_code_sha256": manifest["sha256"].get("code", {}),
              "training_code_matches_current": {relative: digest(ROOT / relative) == expected
                  for relative, expected in manifest["sha256"].get("code", {}).items() if (ROOT / relative).is_file()},
              "model_artifact": "patient_linking.pkl is locally generated trusted pickle; never load untrusted pickles",
              "ner_inference_performed": False, "old_cpu_experiment_rerun": False, "target_data_read": False}
    if frozen_path is not None:
        config["frozen_selection"] = {"path": str(paths["frozen_selection"]), "sha256": input_hashes["frozen_selection"],
                                      "selected": {"entity_strategy": strategies[0], "association_threshold": association_thresholds[0],
                                                   "span_threshold": threshold},
                                      "declared_development_fold": 0, "current_fold_parameter_selection": False}
    if any(digest(path) != input_hashes[name] for name, path in paths.items()):
        raise RuntimeError("An input artifact changed during evaluation.")
    if any(digest(ROOT / relative) != expected for relative, expected in code_hashes.items()):
        raise RuntimeError("Evaluation source changed during this run.")
    config["artifact_sha256"] = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
    write_json(output / "config.json", config)
    return {"output_dir": str(output), "selected": selected, "valid": validation_report["valid"], "scope": scope}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--umls-terms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl")
    parser.add_argument("--ontology", type=Path, default=ROOT / "PatientPheX-V1-A/hp.obo")
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--frozen-selection", type=Path,
                        help="JSON with selected entity_strategy, association_threshold and span_threshold frozen from development fold 0; evaluate one configuration only")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
