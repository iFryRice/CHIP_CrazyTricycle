"""Test complementary BIO evidence, distinct from replacing the NER pipeline."""

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
from scripts.evaluate_semantic_linking import verify_upstream_partition
from scripts.optimize_cpu_association import blind, from_scores, validate_file


def run():
    started = time.time()
    path = ROOT / "experiments/bio_corroboration/plan.json"
    plan = json.loads(path.read_text())
    for filename, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if digest_file(ROOT / filename) != expected:
            raise ValueError(f"Changed input/code: {filename}")
    report = ROOT / plan["report_dir"]
    if report.exists():
        raise FileExistsError("Refusing to overwrite an earlier corroboration pilot.")
    documents = read_jsonl(ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl")
    index = {str(d["pmc_id"]): d for d in documents}
    splits = json.loads((ROOT / "reports/cpu_baseline/split.json").read_text())["folds"]
    baseline_index = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / "reports/association_context/best_development_oof.jsonl")}
    report.mkdir(parents=True)
    write_json(report / "plan.json", plan)
    predictions = {config["name"]: [] for config in plan["configurations"]}
    fold_results, pilot_docs = [], []
    for fold in plan["pilot_folds"]:
        validation = [index[key] for key in splits[fold]]
        pilot_docs.extend(validation)
        directory = ROOT / f"work/span_ner/bio-pilot-fold{fold}"
        manifest = json.loads((directory / "manifest.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        verify_upstream_partition(manifest, splits[fold], index, set())
        if summary["status"] != "completed" or summary["epochs_completed"] != 5 or manifest["fit_all"] or manifest["smoke"]:
            raise ValueError("BIO training did not complete the fixed fold protocol.")
        source_records = read_jsonl(directory / "last_validation_spans.jsonl")
        raw = {str(d["pmc_id"]): d["spans"] for d in source_records}
        if set(raw) != set(splits[fold]):
            raise ValueError("BIO span coverage differs from the validation fold.")
        candidates = json.loads((ROOT / f"reports/bio_ner/pilot/fold{fold}/candidates.json").read_text())
        with (ROOT / f"work/span_ner/document_abbreviations/fold{fold}.pkl").open("rb") as handle:
            model = pickle.load(handle)
        reproduced = [from_scores(d, baseline_index[str(d["pmc_id"])]["entities"],
                        model.predict_scores(blind(d), baseline_index[str(d["pmc_id"])]["entities"]), 0.5) for d in validation]
        if reproduced != [baseline_index[str(d["pmc_id"])] for d in validation]:
            raise ValueError("The baseline patient model failed exact reproduction.")
        base_metrics = evaluate(validation, reproduced)
        row = {"fold": fold, "baseline": base_metrics, "configurations": {}}
        for config in plan["configurations"]:
            local, edits = [], {}
            for document in validation:
                identifier = str(document["pmc_id"])
                entities, edits[identifier] = revise_entities(baseline_index[identifier]["entities"], raw[identifier], candidates[identifier],
                    veto_acronyms=config["veto_acronyms"], add_mentions=config["add_mentions"])
                local.append(from_scores(document, entities, model.predict_scores(blind(document), entities), 0.5))
            predictions[config["name"]].extend(local)
            row["configurations"][config["name"]] = evaluate(validation, local)
            write_json(report / f"fold{fold}_{config['name']}_edits.json", edits)
            print(f"fold={fold} config={config['name']} score={row['configurations'][config['name']]['score']:.6f} gain={row['configurations'][config['name']]['score']-base_metrics['score']:.6f}", flush=True)
        fold_results.append(row)
        write_json(report / f"fold{fold}_metrics.json", row)
    ranked = []
    for config in plan["configurations"]:
        deltas = [row["configurations"][config["name"]]["score"] - row["baseline"]["score"] for row in fold_results]
        mean_gain = sum(deltas) / len(deltas)
        checks = {"mean_gain": mean_gain >= plan["promotion"]["minimum_mean_gain"],
                  "worst_fold": min(deltas) >= -plan["promotion"]["maximum_fold_loss"]}
        ranked.append({"configuration": config, "mean_gain": mean_gain, "fold_deltas": deltas,
                       "metrics": evaluate(pilot_docs, predictions[config["name"]]), "checks": checks, "eligible": all(checks.values())})
    ranked.sort(key=lambda row: (-row["mean_gain"], row["configuration"]["name"]))
    eligible = [row for row in ranked if row["eligible"]]
    best = eligible[0] if eligible else ranked[0]
    write_jsonl(report / "best_pilot_predictions.jsonl", predictions[best["configuration"]["name"]])
    validate_file(report / "best_pilot_predictions.jsonl", pilot_docs, Ontology(ROOT / "PatientPheX-V1-A/hp.obo"), report / "validation.json")
    result = {"plan_sha256": digest_file(path), "ranked": ranked, "selected": best["configuration"] if eligible else None,
              "elapsed_seconds": time.time() - started, "scope": "Two reused development folds only; no B output. Advance a passing frozen configuration to the remaining three folds."}
    write_json(report / "summary.json", result)
    print(json.dumps({"selected": result["selected"], "mean_gain": best["mean_gain"]}), flush=True)


if __name__ == "__main__":
    run()
