"""Compare four frozen document-definition policies on existing outer folds."""

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.abbreviations import apply_definitions, resolve_definitions
from patientphex.context_linking import ContextTreeLinker
from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.span_linking import SpanLinker
from scripts.optimize_cpu_association import blind, bootstrap_delta, error_counts, from_scores, validate_file


def promotion_checks(base, metrics, deltas, gates):
    return {"minimum_gain": metrics["score"] - base["score"] >= gates["minimum_gain"],
            "all_components_nonworse": all(metrics[key]["f1"] >= base[key]["f1"] for key in ("mention", "document", "association_micro", "association_macro")),
            "fold_consistency": sum(delta >= 0 for delta in deltas) >= gates["minimum_nonworse_folds"],
            "worst_fold": min(deltas) >= -gates["maximum_fold_loss"],
            "precision_guard": all(metrics[key]["precision"] >= base[key]["precision"] - gates["maximum_precision_loss"] for key in ("mention", "association_micro"))}


def run(plan_path):
    started = time.time()
    plan = json.loads(plan_path.read_text())
    for path, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if digest_file(ROOT / path) != expected:
            raise ValueError(f"Frozen input/code mismatch: {path}")
    report, work, output = ROOT / plan["report_dir"], ROOT / plan["work_dir"], ROOT / plan["output"]
    if report.exists() or work.exists() or output.exists():
        raise FileExistsError("Refusing to replace earlier abbreviation artifacts.")
    documents, target = read_jsonl(ROOT / plan["train_path"]), read_jsonl(ROOT / plan["target_path"])
    index = {str(d["pmc_id"]): d for d in documents}
    target_ids = {str(d["pmc_id"]) for d in target}
    splits = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    if len(documents) != 80 or sorted(sum(splits, [])) != sorted(index) or set(index) & target_ids:
        raise ValueError("The outer partition or data changed.")
    if len(target) != 100 or sum(len(d["patient"]) for d in target) != 244 or any(d.get("entities") or d.get("association") for d in target):
        raise ValueError("Expected complete blind B inputs.")
    baseline = read_jsonl(ROOT / plan["baseline_oof"])
    base_index = {str(d["pmc_id"]): d for d in baseline}
    base_metrics = evaluate(documents, baseline)
    selected = json.loads((ROOT / "reports/association_context/summary.json").read_text())["selected"]
    if selected != {"name": "leaves7_t0.5", "leaves": 7, "threshold": 0.5}:
        raise ValueError("The upstream association configuration changed.")
    report.mkdir(parents=True)
    work.mkdir(parents=True)
    write_json(report / "plan.json", plan)
    ontology = Ontology(ROOT / plan["ontology_path"])
    variants = {config["name"]: [] for config in plan["configurations"]}
    folds = []
    for fold, val_ids in enumerate(splits):
        training = [d for d in documents if str(d["pmc_id"]) not in val_ids]
        validation = [index[identifier] for identifier in val_ids]
        source = ROOT / f"reports/cpu_crossfit/fold{fold}"
        candidates = {str(d["pmc_id"]): d["entities"] for d in read_jsonl(source / "training_candidates.jsonl")}
        provenance = json.loads((source / "candidate_provenance.json").read_text())
        model = ContextTreeLinker(selected["leaves"]).fit(training, candidates, provenance=provenance, forbidden_document_ids=set(val_ids) | target_ids)
        model_path = work / f"fold{fold}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        reproduced = [from_scores(d, base_index[str(d["pmc_id"])]["entities"],
                       model.predict_scores(blind(d), base_index[str(d["pmc_id"])]["entities"]), selected["threshold"]) for d in validation]
        if reproduced != [base_index[str(d["pmc_id"])] for d in validation]:
            raise ValueError("The unchanged association baseline did not reproduce exactly.")
        linker = SpanLinker(ontology, ROOT / plan["umls_path"]).fit(training, forbidden_document_ids=set(val_ids) | target_ids)
        definitions = {str(d["pmc_id"]): resolve_definitions(blind(d), linker) for d in validation}
        fold_dir = report / f"fold{fold}"
        fold_dir.mkdir()
        write_json(fold_dir / "definitions.json", definitions)
        write_json(fold_dir / "provenance.json", {"linker": linker.fit_audit, "association": model.provenance_, "association_model_sha256": digest_file(model_path), "baseline_reproduced": True})
        row = {"fold": fold, "baseline": evaluate(validation, reproduced), "configurations": {}, "changes": {}}
        for config in plan["configurations"]:
            predictions, edits = [], {}
            counts = Counter()
            for document in validation:
                identifier = str(document["pmc_id"])
                entities, actions = apply_definitions(blind(document), base_index[identifier]["entities"], definitions[identifier],
                    add_missing=config["add_missing"], suppress_unresolved=config["suppress_unresolved"])
                predictions.append(from_scores(document, entities, model.predict_scores(blind(document), entities), selected["threshold"]))
                edits[identifier] = actions
                counts.update(action["action"] for action in actions)
            variants[config["name"]].extend(predictions)
            metrics = evaluate(validation, predictions)
            row["configurations"][config["name"]] = metrics
            row["changes"][config["name"]] = dict(counts)
            write_json(fold_dir / f"{config['name']}_edits.json", edits)
            print(f"fold={fold} policy={config['name']} score={metrics['score']:.6f} gain={metrics['score']-row['baseline']['score']:.6f} edits={dict(counts)}", flush=True)
        folds.append(row)
        write_json(fold_dir / "metrics.json", row)
    ranked = []
    for config in plan["configurations"]:
        predictions = variants[config["name"]]
        metrics = evaluate(documents, predictions)
        deltas = [row["configurations"][config["name"]]["score"] - row["baseline"]["score"] for row in folds]
        checks = promotion_checks(base_metrics, metrics, deltas, plan["promotion"])
        ranked.append({"configuration": config, "metrics": metrics, "fold_deltas": deltas, "checks": checks,
                       "eligible": all(checks.values()), "errors": error_counts(documents, predictions)})
    ranked.sort(key=lambda row: (-row["metrics"]["score"], row["configuration"]["name"]))
    eligible = [row for row in ranked if row["eligible"]]
    chosen = eligible[0] if eligible else None
    best = chosen or ranked[0]
    write_jsonl(report / "best_development_oof.jsonl", variants[best["configuration"]["name"]])
    validate_file(report / "best_development_oof.jsonl", documents, ontology, report / "oof_validation.json")
    summary = {"plan_sha256": digest_file(plan_path), "baseline": base_metrics, "ranked": ranked,
               "selected": chosen["configuration"] if chosen else None,
               "bootstrap": bootstrap_delta(documents, baseline, variants[best["configuration"]["name"]]),
               "scope": "Repeated development CV; this experiment does not measure official B accuracy."}
    if chosen:
        config = chosen["configuration"]
        with (ROOT / plan["association_model"]).open("rb") as handle:
            model = pickle.load(handle)
        linker = SpanLinker(ontology, ROOT / plan["umls_path"]).fit(documents, forbidden_document_ids=target_ids)
        b_original = read_jsonl(ROOT / plan["baseline_b"])
        b_index = {str(d["pmc_id"]): d for d in b_original}
        reproduced = [from_scores(d, b_index[str(d["pmc_id"])]["entities"],
                        model.predict_scores(blind(d), b_index[str(d["pmc_id"])]["entities"]), selected["threshold"]) for d in target]
        if reproduced != b_original:
            raise ValueError("The frozen B association model no longer reproduces the baseline.")
        predictions, edits, definitions = [], {}, {}
        for document in target:
            identifier = str(document["pmc_id"])
            definitions[identifier] = resolve_definitions(blind(document), linker)
            entities, edits[identifier] = apply_definitions(blind(document), b_index[identifier]["entities"], definitions[identifier],
                add_missing=config["add_missing"], suppress_unresolved=config["suppress_unresolved"])
            predictions.append(from_scores(document, entities, model.predict_scores(blind(document), entities), selected["threshold"]))
        write_jsonl(output, predictions)
        write_json(report / "b_definitions.json", definitions)
        write_json(report / "b_edits.json", edits)
        write_json(report / "final_linker_provenance.json", linker.fit_audit)
        summary["b_validation"] = validate_file(output, target, ontology, report / "b_validation.json")
        summary["output_sha256"] = digest_file(output)
    summary["elapsed_seconds"] = time.time() - started
    write_json(report / "summary.json", summary)
    print(json.dumps({"selected": summary["selected"], "best_score": best["metrics"]["score"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/document_abbreviations/plan.json")
    run(parser.parse_args().plan)
