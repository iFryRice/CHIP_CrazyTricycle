"""Evaluate frozen structural tree models without refitting entity extraction."""

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.context_linking import ContextTreeLinker
from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from scripts.optimize_cpu_association import blind, bootstrap_delta, error_counts, from_scores, promotion_checks, validate_file


def run(plan_path):
    started = time.time()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for path, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if digest_file(ROOT / path) != expected:
            raise ValueError(f"Frozen artifact mismatch: {path}")
    report, output = ROOT / plan["report_dir"], ROOT / plan["output"]
    if (report / "plan.json").exists() or output.exists():
        raise FileExistsError("Refusing to overwrite an earlier experiment.")
    report.mkdir(exist_ok=True, parents=True)
    write_json(report / "plan.json", plan)
    documents, target = read_jsonl(ROOT / plan["train_path"]), read_jsonl(ROOT / plan["target_path"])
    index = {str(d["pmc_id"]): d for d in documents}
    target_ids = {str(d["pmc_id"]) for d in target}
    splits = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    if len(documents) != 80 or sorted(sum(splits, [])) != sorted(index) or set(index) & target_ids:
        raise ValueError("The outer document partition changed.")
    if len(target) != 100 or sum(len(d["patient"]) for d in target) != 244 or any(d.get("entities") or d.get("association") for d in target):
        raise ValueError("Expected complete blind B targets.")
    baseline = read_jsonl(ROOT / plan["baseline_oof"])
    base_index = {str(d["pmc_id"]): d for d in baseline}
    base_metrics = evaluate(documents, baseline)
    variants = {config["name"]: [] for config in plan["configurations"]}
    folds = []
    for fold, val_ids in enumerate(splits):
        training = [d for d in documents if str(d["pmc_id"]) not in val_ids]
        validation = [index[identifier] for identifier in val_ids]
        source = ROOT / f"reports/cpu_crossfit/fold{fold}"
        candidates = {str(d["pmc_id"]): d["entities"] for d in read_jsonl(source / "training_candidates.jsonl")}
        provenance = json.loads((source / "candidate_provenance.json").read_text())
        fold_dir = report / f"fold{fold}"
        fold_dir.mkdir()
        row = {"fold": fold, "baseline": evaluate(validation, [base_index[str(d["pmc_id"])] for d in validation]), "configurations": {}}
        for leaves in sorted({config["leaves"] for config in plan["configurations"]}):
            model = ContextTreeLinker(leaves).fit(training, candidates, provenance=provenance,
                                                 forbidden_document_ids=set(val_ids) | target_ids)
            write_json(fold_dir / f"training_leaves{leaves}.json", model.diagnostics_)
            scores = {str(d["pmc_id"]): model.predict_scores(blind(d), base_index[str(d["pmc_id"])]["entities"]) for d in validation}
            write_json(fold_dir / f"scores_leaves{leaves}.json", scores)
            for config in plan["configurations"]:
                if config["leaves"] != leaves:
                    continue
                predictions = [from_scores(d, base_index[str(d["pmc_id"])]["entities"], scores[str(d["pmc_id"])], config["threshold"]) for d in validation]
                variants[config["name"]].extend(predictions)
                metrics = evaluate(validation, predictions)
                row["configurations"][config["name"]] = metrics
                print(f"fold={fold} config={config['name']} score={metrics['score']:.6f} gain={metrics['score']-row['baseline']['score']:.6f}", flush=True)
        folds.append(row)
        write_json(fold_dir / "metrics.json", row)
    ranked = []
    for config in plan["configurations"]:
        predictions = variants[config["name"]]
        metrics = evaluate(documents, predictions)
        deltas = [row["configurations"][config["name"]]["score"] - row["baseline"]["score"] for row in folds]
        checks = promotion_checks(base_metrics, metrics, deltas, plan["promotion"])
        ranked.append({"configuration": config, "metrics": metrics, "fold_deltas": deltas,
                       "checks": checks, "eligible": all(checks.values()), "errors": error_counts(documents, predictions)})
    ranked.sort(key=lambda row: (-row["metrics"]["score"], row["configuration"]["name"]))
    eligible = [row for row in ranked if row["eligible"]]
    selected = eligible[0] if eligible else None
    best = selected or ranked[0]
    ontology = Ontology(ROOT / plan["ontology_path"])
    write_jsonl(report / "best_development_oof.jsonl", variants[best["configuration"]["name"]])
    validate_file(report / "best_development_oof.jsonl", documents, ontology, report / "oof_validation.json")
    summary = {"plan_sha256": digest_file(plan_path), "baseline": base_metrics, "ranked": ranked,
               "selected": selected["configuration"] if selected else None,
               "bootstrap": bootstrap_delta(documents, baseline, variants[best["configuration"]["name"]]),
               "scope": "Repeated development CV, not independent test or official B accuracy."}
    if selected:
        final_source = read_jsonl(ROOT / "reports/cpu_b_control/out_of_fold_predictions.jsonl")
        candidates = {str(d["pmc_id"]): d["entities"] for d in final_source}
        by_document = {}
        for val_ids in splits:
            sources = sorted(set(index) - set(val_ids))
            for identifier in val_ids:
                by_document[identifier] = {"training_document_ids": sources, "label_source_document_ids": sources}
        config = selected["configuration"]
        model = ContextTreeLinker(config["leaves"]).fit(documents, candidates,
                provenance={"kind": "out_of_fold", "by_document": by_document}, forbidden_document_ids=target_ids)
        work = ROOT / "work/span_ner/association_context"
        work.mkdir(parents=True, exist_ok=False)
        with (work / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        write_json(report / "final_training.json", {"diagnostics": model.diagnostics_, "provenance": model.provenance_, "model_sha256": digest_file(work / "model.pkl")})
        b_index = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["baseline_b"])}
        predictions = [from_scores(d, b_index[str(d["pmc_id"])]["entities"],
                        model.predict_scores(blind(d), b_index[str(d["pmc_id"])]["entities"]), config["threshold"]) for d in target]
        write_jsonl(output, predictions)
        summary["b_validation"] = validate_file(output, target, ontology, report / "b_validation.json")
        summary["output_sha256"] = digest_file(output)
    summary["elapsed_seconds"] = time.time() - started
    write_json(report / "summary.json", summary)
    print(json.dumps({"selected": summary["selected"], "best_score": best["metrics"]["score"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/association_context/plan.json")
    run(parser.parse_args().plan)
