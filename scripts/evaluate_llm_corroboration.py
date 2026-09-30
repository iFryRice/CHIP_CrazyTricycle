"""Test frozen LLM support policies on the already completed two-fold pilot."""

import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.llm_corroboration import filter_entities
from patientphex.ontology import Ontology
from patientphex.patient_linking import _check_provenance
from scripts.evaluate_semantic_linking import verify_upstream_partition
from scripts.optimize_cpu_association import blind, bootstrap_delta, error_counts, from_scores, validate_file


def run():
    started = time.time()
    path = ROOT / "experiments/llm_corroboration/plan.json"
    plan = json.loads(path.read_text())
    for name, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"Frozen input/code changed: {name}")
    report = ROOT / plan["report_dir"]
    if report.exists():
        raise FileExistsError("Refusing to overwrite another LLM corroboration pilot.")
    documents = read_jsonl(ROOT / plan["train_path"])
    lookup = {str(document["pmc_id"]): document for document in documents}
    splits = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    baseline = {str(document["pmc_id"]): document for document in read_jsonl(ROOT / plan["baseline_oof"])}
    report.mkdir(parents=True)
    write_json(report / "plan.json", plan)
    outputs = {config["name"]: [] for config in plan["configurations"]}
    rows, pilot_documents, pilot_baseline = [], [], []
    for fold in plan["pilot_folds"]:
        directory = ROOT / plan["inference_work_dir"] / f"fold{fold}"
        summary = json.loads((directory / "summary.json").read_text())
        verify_upstream_partition(summary, splits[fold], lookup, set())
        if summary["status"] != "completed" or summary["plan_sha256"] != digest_file(ROOT / plan["inference_plan"]):
            raise ValueError("LLM inference is incomplete or comes from another frozen plan.")
        if summary["format_success_rate"] < 0.98 or digest_file(directory / "spans.jsonl") != summary["span_sha256"]:
            raise ValueError("LLM output format or content fingerprint failed.")
        raw = {str(record["pmc_id"]): record["spans"] for record in read_jsonl(directory / "spans.jsonl")}
        if set(raw) != set(splits[fold]):
            raise ValueError("LLM span coverage differs from the fixed fold.")
        with (ROOT / f"work/span_ner/document_abbreviations/fold{fold}.pkl").open("rb") as handle:
            model = pickle.load(handle)
        _check_provenance(model.provenance_, set(lookup) - set(splits[fold]), set(splits[fold]))
        validation = [lookup[identifier] for identifier in splits[fold]]
        original = [baseline[str(document["pmc_id"])] for document in validation]
        replay = [from_scores(document, baseline[str(document["pmc_id"])]["entities"],
                  model.predict_scores(blind(document), baseline[str(document["pmc_id"])]["entities"]), 0.5) for document in validation]
        if replay != original:
            raise ValueError("Frozen patient-linker baseline failed exact replay.")
        pilot_documents.extend(validation)
        pilot_baseline.extend(original)
        row = {"fold": fold, "baseline": evaluate(validation, original), "configurations": {}}
        for config in plan["configurations"]:
            local, edits = [], {}
            for document in validation:
                identifier = str(document["pmc_id"])
                entities, edits[identifier] = filter_entities(baseline[identifier]["entities"], raw[identifier], config["scope"])
                local.append(from_scores(document, entities, model.predict_scores(blind(document), entities), 0.5))
            outputs[config["name"]].extend(local)
            metrics = evaluate(validation, local)
            row["configurations"][config["name"]] = metrics
            write_json(report / f"fold{fold}_{config['name']}_edits.json", edits)
            print(json.dumps({"fold": fold, "configuration": config["name"], "score": metrics["score"],
                              "gain": metrics["score"]-row["baseline"]["score"],
                              "removed_entities": sum(len(value["removed"]) for value in edits.values())}), flush=True)
        rows.append(row)
        write_json(report / f"fold{fold}_metrics.json", row)
    baseline_metrics = evaluate(pilot_documents, pilot_baseline)
    ranked, gate = [], plan["promotion"]
    for config in plan["configurations"]:
        predicted = outputs[config["name"]]
        metrics = evaluate(pilot_documents, predicted)
        deltas = [row["configurations"][config["name"]]["score"] - row["baseline"]["score"] for row in rows]
        mean_gain = sum(deltas) / len(deltas)
        checks = {"mean_gain": mean_gain >= gate["minimum_mean_gain"], "worst_fold": min(deltas) >= -gate["maximum_fold_loss"],
                  "component_guard": all(metrics[name]["f1"] >= baseline_metrics[name]["f1"] - gate["maximum_component_loss"]
                                         for name in ["mention", "document", "association_micro", "association_macro"])}
        ranked.append({"configuration": config, "metrics": metrics, "mean_gain": mean_gain, "fold_deltas": deltas,
                       "checks": checks, "eligible": all(checks.values()), "errors": error_counts(pilot_documents, predicted)})
    ranked.sort(key=lambda row: (-row["mean_gain"], row["configuration"]["name"]))
    eligible = [row for row in ranked if row["eligible"]]
    best = eligible[0] if eligible else ranked[0]
    predicted = outputs[best["configuration"]["name"]]
    write_jsonl(report / "best_pilot_predictions.jsonl", predicted)
    validate_file(report / "best_pilot_predictions.jsonl", pilot_documents, Ontology(ROOT / plan["ontology_path"]), report / "validation.json")
    result = {"plan_sha256": digest_file(path), "baseline": baseline_metrics, "ranked": ranked,
              "selected": best["configuration"] if eligible else None, "best_bootstrap": bootstrap_delta(pilot_documents, pilot_baseline, predicted),
              "elapsed_seconds": time.time()-started,
              "scope": "New predeclared filtering hypothesis on two reused development folds, using fixed source-exact Qwen spans. Remaining-fold confirmation is mandatory before a B candidate."}
    write_json(report / "summary.json", result)
    print(json.dumps({"selected": result["selected"], "best_mean_gain": best["mean_gain"]}), flush=True)


if __name__ == "__main__":
    run()
