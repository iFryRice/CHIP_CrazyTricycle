"""Audit saved span predictions and diagnose validation-selected thresholds.

This CPU/stdlib analysis never loads a checkpoint or performs model inference.
Scores below the saved prediction threshold are unavailable and cannot be tested.
All reported optima are validation-selected diagnostics, not independent tests.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patientphex.data import read_jsonl
from scripts.train_span_ner import gold_span_sets, score_predictions

LABELS = ("positive", "NO")
DEFAULT_THRESHOLDS = (0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 1000 / 1001, 0.9995, 0.9999)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filter_scores(predictions: dict, thresholds: tuple[float, float]) -> dict:
    return {key: score for key, score in predictions.items() if score >= thresholds[key[3]]}


def check_metric_equal(actual: dict, expected: dict) -> None:
    for group in (*LABELS, "micro", "any_span"):
        for name in ("tp", "fp", "fn", "predicted", "gold", "precision", "recall", "f1"):
            if not math.isclose(actual[group][name], expected[group][name], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Saved-score baseline does not reproduce summary: {group}.{name}")


def exact_best_threshold(predictions: dict, gold: list[set], minimum: float, label: int | None = None) -> dict:
    gold_keys = {(*key, current) for current, values in enumerate(gold) for key in values
                 if label is None or current == label}
    ordered = sorted(((score, key in gold_keys) for key, score in predictions.items()
                      if label is None or key[3] == label), reverse=True)
    # Descending groups preserve >= semantics, including all tied scores.
    candidates = []
    count = true_positive = 0
    for threshold, group in itertools.groupby(ordered, key=lambda item: item[0]):
        group = list(group)
        count += len(group)
        true_positive += sum(hit for _, hit in group)
        denominator = count + len(gold_keys)
        candidates.append({"threshold": threshold, "f1": 2 * true_positive / denominator if denominator else 0.0,
                           "predicted": count, "tp": true_positive})
    if not ordered or ordered[0][0] < 1.0:
        candidates.append({"threshold": 1.0, "f1": 0.0, "predicted": 0, "tp": 0})
    denominator = len(ordered) + len(gold_keys)
    candidates.append({"threshold": minimum, "f1": 2 * true_positive / denominator if denominator else 0.0,
                       "predicted": len(ordered), "tp": true_positive})
    best = max(candidates, key=lambda point: (point["f1"], point["threshold"]))
    return {**best, "distinct_score_thresholds_checked": len({point["threshold"] for point in candidates}),
            "tie_break": "Choose the higher threshold when F1 ties"}


def false_positive_profile(predictions: dict, gold: list[set]) -> dict:
    by_document = {}
    for label, values in enumerate(gold):
        for pmc_id, offset, length in values:
            by_document.setdefault(pmc_id, []).append((offset, offset + length, label))
    result = {key: 0 for key in ("same_span_wrong_label", "overlaps_gold_same_label",
                               "overlaps_gold_other_label", "no_overlap_with_any_gold")}
    for (pmc_id, offset, length, label), _ in predictions.items():
        if (pmc_id, offset, length) in gold[label]:
            continue
        overlaps = [item for item in by_document.get(pmc_id, []) if offset < item[1] and offset + length > item[0]]
        if (pmc_id, offset, length) in gold[1 - label]:
            category = "same_span_wrong_label"
        elif any(item[2] == label for item in overlaps):
            category = "overlaps_gold_same_label"
        elif overlaps:
            category = "overlaps_gold_other_label"
        else:
            category = "no_overlap_with_any_gold"
        result[category] += 1
    return {"counts": result, "definition": "Exclusive diagnostic categories; overlapping but inexact spans remain false positives"}


def analyze(run_dir: Path, data_path: Path, thresholds: list[float]) -> dict:
    paths = {name: run_dir / filename for name, filename in (
        ("manifest", "manifest.json"), ("summary", "summary.json"),
        ("predictions", "best_validation_predictions.jsonl"),
    )}
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    minimum = float(manifest["parameters"]["threshold"])
    if minimum != 0.5 or summary["threshold"] != minimum:
        raise ValueError("This audit requires the saved baseline threshold to be exactly 0.5")
    if any(not math.isfinite(value) or not minimum <= value <= 1.0 for value in thresholds):
        raise ValueError("Thresholds must lie in [0.5, 1]; lower scores were not saved")
    if summary["best_epoch"] != summary["epochs_completed"]:
        raise ValueError("Summary contains last-epoch metrics but saved predictions are from a different best epoch")
    if file_hash(data_path) != manifest["sha256"]["data"]:
        raise ValueError("Training data hash differs from the completed run")
    for relative, expected in manifest["sha256"]["code"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"Code hash differs from completed run: {relative}")
    documents = read_jsonl(data_path)
    validation_ids = {str(item) for item in manifest["validation_document_ids"]}
    training_ids = {str(item) for item in manifest["training_document_ids"]}
    if validation_ids & training_ids or validation_ids | training_ids != {str(doc["pmc_id"]) for doc in documents}:
        raise ValueError("Manifest document IDs are not a disjoint complete partition of the input")
    validation = [doc for doc in documents if str(doc["pmc_id"]) in validation_ids]
    sources = {str(doc["pmc_id"]): doc for doc in validation}
    records = read_jsonl(paths["predictions"])
    if {str(record["pmc_id"]) for record in records} != validation_ids:
        raise ValueError("Prediction documents do not exactly match manifest validation IDs")
    predictions = {}
    for record in records:
        pmc_id = str(record["pmc_id"])
        seen_spans = set()
        for span in record["spans"]:
            offset, length = span["offset"], span["length"]
            if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0:
                raise ValueError("Malformed saved prediction offset or length")
            if (offset, length) in seen_spans:
                raise ValueError("Duplicate span in saved predictions")
            seen_spans.add((offset, length))
            if not any(paragraph["offset"] <= offset and offset + length <= paragraph["offset"] + len(paragraph["text"])
                       and paragraph["text"][offset - paragraph["offset"]:offset - paragraph["offset"] + length] == span["text"]
                       for paragraph in sources[pmc_id]["full_text"]):
                raise ValueError("Saved prediction text does not match original source offsets")
            labels = span["labels"]
            if not labels or len(labels) != len(set(labels)) or set(labels) != set(span["scores"]) or not set(labels) <= set(LABELS):
                raise ValueError("Malformed saved prediction labels/scores")
            for label in labels:
                value = span["scores"][label]
                if not isinstance(value, (float, int)) or not math.isfinite(value) or not minimum <= value <= 1.0:
                    raise ValueError("Saved score is non-finite or outside the retained threshold range")
                predictions[(pmc_id, offset, length, LABELS.index(label))] = value
    baseline = score_predictions(predictions, validation)
    check_metric_equal(baseline, summary["last_metrics"])
    gold = gold_span_sets(validation)
    table = [{"threshold": threshold, "metrics": score_predictions(filter_scores(predictions, (threshold, threshold)), validation)}
             for threshold in thresholds]
    common = exact_best_threshold(predictions, gold, minimum)
    common_predictions = filter_scores(predictions, (common["threshold"], common["threshold"]))
    common["metrics"] = score_predictions(common_predictions, validation)
    per_label = {}
    for label, name in enumerate(LABELS):
        point = exact_best_threshold(predictions, gold, minimum, label)
        point["metrics"] = score_predictions(filter_scores(predictions, (point["threshold"], point["threshold"])), validation)[name]
        per_label[name] = point
    selected = tuple(per_label[name]["threshold"] for name in LABELS)
    weights = manifest["class_statistics"]["pos_weight"]
    return {
        "scope": "Validation-selected threshold diagnostics for the already validation-selected best checkpoint; NOT independent test performance",
        "inference_performed": False, "training_performed": False, "saved_score_floor": minimum,
        "search_limit": "Exact for thresholds >= 0.5 on saved max-over-window scores; thresholds below 0.5 cannot be reconstructed",
        "validation_document_ids": sorted(validation_ids), "best_epoch": summary["best_epoch"],
        "provenance": {"input_sha256": {name: file_hash(path) for name, path in paths.items()},
                       "data_sha256": file_hash(data_path), "analysis_code_sha256": file_hash(Path(__file__)),
                       "runner_code_hash_matches_manifest": True},
        "baseline_reproduces_summary": True, "baseline": baseline, "thresholds_evaluated": thresholds,
        "common_threshold_table": table, "best_common_threshold": common,
        "independent_per_label_f1_optima": per_label,
        "combined_independent_label_thresholds": {
            "thresholds": dict(zip(LABELS, selected)),
            "metrics": score_predictions(filter_scores(predictions, selected), validation),
            "note": "Individual label-F1 optima; not a joint micro-F1 threshold optimization. NO has only 30 validation gold spans in this run.",
        },
        "false_positive_profiles": {"baseline": false_positive_profile(predictions, gold),
                                    "best_common_threshold": false_positive_profile(common_predictions, gold)},
        "weighted_bce_interpretation": {
            "actual_class_statistics": manifest["class_statistics"],
            "formula": "Ideal weighted-BCE optimum s = w*q/(1-q+w*q); equivalently logit(s)=logit(q)+log(w)",
            "score_0_5_implied_unweighted_posterior": dict(zip(LABELS, [1 / (1 + weight) for weight in weights])),
            "unweighted_posterior_0_5_implied_score": dict(zip(LABELS, [weight / (1 + weight) for weight in weights])),
            "normalizer_effect": "A positive per-label constant rescales gradients; it does not undo pos_weight or calibrate sigmoid outputs",
            "limitation": "Idealized loss argument, not proof of empirical calibration. Finite model fitting, per-batch averaging and max-over-window decoding can change calibration.",
        },
        "interpretation": "Raising the threshold removes many false positives, but even the selected optimum retains substantial false positives. Diagnose ranking and span boundaries before changing training.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports/span_ner/fold0")
    parser.add_argument("--data", type=Path, default=ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace this diagnostic report only")
    args = parser.parse_args()
    output = args.output or args.run_dir / "threshold_diagnostics.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Diagnostic report already exists: {output}; use --overwrite explicitly")
    if output.resolve() in {(args.run_dir / name).resolve() for name in ("manifest.json", "summary.json", "best_validation_predictions.jsonl")} or output.resolve() == args.data.resolve():
        raise ValueError("Output must not overwrite an input artifact")
    result = analyze(args.run_dir, args.data, args.thresholds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "baseline_micro": result["baseline"]["micro"],
                      "selected_threshold": result["best_common_threshold"]["threshold"],
                      "selected_validation_micro": result["best_common_threshold"]["metrics"]["micro"],
                      "scope": result["scope"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
