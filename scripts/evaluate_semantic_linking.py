"""Evaluate fixed SapBERT mapping of held-out neural spans, then predict B if qualified."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.association import AssociationModel
from patientphex.data import digest_file, prediction_record, read_jsonl, write_json, write_jsonl
from patientphex.entities import is_negated
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.semantic_linking import SemanticIndex, merge_neural_additions
from patientphex.span_linking import SpanLinker
from scripts.optimize_cpu_association import blind, bootstrap_delta, error_counts, validate_file


def verify_upstream_partition(manifest, validation_ids, all_ids, target_ids):
    train_ids = [str(value) for value in manifest["training_document_ids"]]
    val_ids = [str(value) for value in manifest["validation_document_ids"]]
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError("Duplicate upstream training/validation document IDs.")
    if set(val_ids) != set(validation_ids) or set(train_ids) != set(all_ids) - set(validation_ids):
        raise ValueError("Neural training does not respect the outer document split.")
    if set(train_ids) & (set(validation_ids) | set(target_ids)):
        raise ValueError("Neural training includes held-out or target labels.")


def eligible_spans(record, minimum):
    return [span for span in record["spans"] if span["scores"].get("positive", 0) >= minimum
            and span["scores"].get("positive", 0) >= span["scores"].get("NO", 0)
            and 2 <= len(span["text"]) <= 220 and any(c.isalpha() for c in span["text"])]


def proposals(document, record, linker, retrieved, minimum):
    if str(document["pmc_id"]) != str(record["pmc_id"]):
        raise ValueError("Neural predictions are paired with the wrong source.")
    result = []
    for span in eligible_spans(record, minimum):
        matching = [p for p in document["full_text"] if p["offset"] <= span["offset"]
                    and span["offset"] + span["length"] <= p["offset"] + len(p["text"])]
        if not matching:
            raise ValueError("Neural span falls outside original source paragraphs.")
        paragraph = matching[0]
        start = span["offset"] - paragraph["offset"]
        end = start + span["length"]
        if paragraph["text"][start:end] != span["text"]:
            raise ValueError("Neural span changes original source text.")
        resolution = linker.resolve(span["text"])
        if resolution["identifier"] != "-1":
            identifier, source, similarity, margin = resolution["identifier"], resolution["source"], 1.0, 1.0
        else:
            nearest = retrieved[span["text"]]
            identifier, source = nearest["identifier"], "semantic"
            similarity, margin = nearest["similarity"], nearest["margin"]
        entity = {"identifier": identifier, "type": "Phenotype", "offset": span["offset"],
                  "length": span["length"], "text": span["text"],
                  "note": "NO" if is_negated(paragraph["text"], start, end) else None}
        result.append({"entity": entity, "span_score": span["scores"]["positive"], "source": source,
                       "similarity": similarity, "margin": margin})
    return result


def predict(document, base_entities, candidates, config, margin):
    entities = merge_neural_additions(base_entities, candidates, config["span_threshold"], config["semantic_threshold"], margin)
    association = AssociationModel(mode="nearest").predict(blind(document), entities)
    return prediction_record(document, entities, association)


def run(plan_path):
    started = time.time()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for path, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if digest_file(ROOT / path) != expected:
            raise ValueError(f"Frozen input/code hash mismatch: {path}")
    model_manifest = json.loads((ROOT / plan["model_manifest"]).read_text())
    for filename, expected in model_manifest["files"].items():
        if digest_file(ROOT / plan["model_path"] / filename) != expected["sha256"]:
            raise ValueError(f"Semantic model content mismatch: {filename}")
    report, output = ROOT / plan["report_dir"], ROOT / plan["output"]
    if (report / "plan.json").exists() or output.exists():
        raise FileExistsError("Refusing to overwrite an earlier semantic experiment.")
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "plan.json", plan)
    documents = read_jsonl(ROOT / plan["train_path"])
    targets = read_jsonl(ROOT / plan["target_path"])
    if len(documents) != 80 or len(targets) != 100 or sum(len(d["patient"]) for d in targets) != 244:
        raise ValueError("Unexpected train or B coverage.")
    if any(d.get("entities") or d.get("association") for d in targets):
        raise ValueError("B documents must be blind.")
    document_index = {str(d["pmc_id"]): d for d in documents}
    target_ids = {str(d["pmc_id"]) for d in targets}
    if target_ids & set(document_index):
        raise ValueError("Train and B targets overlap.")
    ontology = Ontology(ROOT / plan["ontology_path"])
    split = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    if sorted(identifier for fold in split for identifier in fold) != sorted(document_index):
        raise ValueError("Outer folds must cover all training documents exactly once.")
    span_sets = []
    minimum = min(config["span_threshold"] for config in plan["configurations"])
    b_manifest = json.loads((ROOT / plan["b_neural_manifest"]).read_text())
    if (b_manifest["role"] != "target" or b_manifest["gold_metrics_computed"]
            or not b_manifest["answers_removed_before_windowing"] or b_manifest["checkpoint_epoch"] != 5
            or b_manifest["score_floor"] > minimum):
        raise ValueError("B candidate provenance or saved threshold is incompatible.")
    if (b_manifest["data_sha256"] != digest_file(ROOT / plan["target_path"])
            or b_manifest["prediction_sha256"] != digest_file(ROOT / plan["b_neural_predictions"])):
        raise ValueError("B candidates do not match their source manifest.")
    for index, source in enumerate(plan["neural_folds"]):
        summary = json.loads((ROOT / source["summary"]).read_text())
        manifest = json.loads((ROOT / source["manifest"]).read_text())
        verify_upstream_partition(manifest, split[index], document_index, target_ids)
        if manifest["sha256"]["data"] != digest_file(ROOT / plan["train_path"]) or manifest["sha256"]["split"] != digest_file(ROOT / plan["split_path"]):
            raise ValueError("Upstream NER data or split hashes differ.")
        if summary.get("best_epoch") != 5 or summary.get("epochs_completed") != 5:
            raise ValueError("Expected predictions from the fixed fifth epoch.")
        records = read_jsonl(ROOT / source["predictions"])
        if {str(d["pmc_id"]) for d in records} != set(split[index]):
            raise ValueError("Neural predictions do not cover their declared held-out fold.")
        span_sets.append(records)
    identity = {"model_manifest_sha256": digest_file(ROOT / plan["model_manifest"]),
                "ontology_sha256": digest_file(ROOT / plan["ontology_path"]),
                "umls_sha256": digest_file(ROOT / plan["umls_path"]),
                "implementation_sha256": digest_file(ROOT / "patientphex/semantic_linking.py")}
    index = SemanticIndex(ROOT / plan["model_path"], ROOT / "work/span_ner/semantic_linking/index", ontology, ROOT / plan["umls_path"], identity)
    witness = index.retrieve(["short stature", "seizures", "hearing loss"])
    expected = {"short stature": "HP:0004322", "seizures": "HP:0001250", "hearing loss": "HP:0000365"}
    if any(witness[text]["identifier"] != identifier for text, identifier in expected.items()):
        raise ValueError(f"Semantic lookup kernel witness failed: {witness}")
    write_json(report / "kernel_witness.json", {"status": "passed", "lookups": witness, "index": index.audit})
    queries = sorted({span["text"] for records in span_sets for record in records for span in eligible_spans(record, minimum)})
    print(f"development_queries={len(queries)}", flush=True)
    retrieved = index.retrieve(queries)
    write_json(ROOT / "work/span_ner/semantic_linking/development_retrieval.json", retrieved)
    original = read_jsonl(ROOT / plan["baseline_oof"])
    original_index = {str(d["pmc_id"]): d for d in original}
    baseline = evaluate(documents, original)
    variants = {config["name"]: [] for config in plan["configurations"]}
    fold_results = []
    for fold, records in enumerate(span_sets):
        val_ids = set(split[fold])
        training = [d for d in documents if str(d["pmc_id"]) not in val_ids]
        validation = [document_index[identifier] for identifier in split[fold]]
        linker = SpanLinker(ontology, ROOT / plan["umls_path"]).fit(training)
        candidates = {str(record["pmc_id"]): proposals(blind(document_index[str(record["pmc_id"])]), record, linker, retrieved, minimum) for record in records}
        fold_dir = report / f"fold{fold}"
        fold_dir.mkdir()
        write_json(fold_dir / "candidates.json", candidates)
        fold_base = evaluate(validation, [original_index[str(d["pmc_id"])] for d in validation])
        row = {"fold": fold, "training_document_ids": sorted(str(d["pmc_id"]) for d in training), "baseline": fold_base, "configurations": {}}
        for config in plan["configurations"]:
            predictions = [predict(d, original_index[str(d["pmc_id"])]["entities"], candidates[str(d["pmc_id"])], config, plan["margin_threshold"]) for d in validation]
            variants[config["name"]].extend(predictions)
            metrics = evaluate(validation, predictions)
            row["configurations"][config["name"]] = metrics
            print(f"fold={fold} config={config['name']} score={metrics['score']:.6f} gain={metrics['score']-fold_base['score']:.6f}", flush=True)
        fold_results.append(row)
        write_json(fold_dir / "metrics.json", row)
    ranked = []
    for config in plan["configurations"]:
        predictions = variants[config["name"]]
        metrics = evaluate(documents, predictions)
        deltas = [row["configurations"][config["name"]]["score"] - row["baseline"]["score"] for row in fold_results]
        gates = plan["promotion"]
        checks = {"minimum_gain": metrics["score"] - baseline["score"] >= gates["minimum_gain"],
                  "fold_consistency": sum(delta >= 0 for delta in deltas) >= gates["minimum_nonworse_folds"],
                  "mention_improved": metrics["mention"]["f1"] > baseline["mention"]["f1"],
                  "document_improved": metrics["document"]["f1"] > baseline["document"]["f1"],
                  "association_guard": all(metrics[m]["f1"] >= baseline[m]["f1"] - gates["maximum_association_f1_loss"] for m in ("association_micro", "association_macro"))}
        ranked.append({"configuration": config, "metrics": metrics, "fold_deltas": deltas, "checks": checks, "eligible": all(checks.values()), "errors": error_counts(documents, predictions)})
    ranked.sort(key=lambda row: (-row["metrics"]["score"], row["configuration"]["name"]))
    eligible = [row for row in ranked if row["eligible"]]
    selected = eligible[0] if eligible else None
    best = selected or ranked[0]
    best_predictions = variants[best["configuration"]["name"]]
    summary = {"plan_sha256": digest_file(plan_path), "baseline": baseline, "ranked": ranked,
               "selected": selected["configuration"] if selected else None, "index": index.audit,
               "best_bootstrap": bootstrap_delta(documents, original, best_predictions),
               "scope": "Repeated fivefold development; not independent testing or official B score."}
    write_jsonl(report / "best_development_oof.jsonl", best_predictions)
    validate_file(report / "best_development_oof.jsonl", documents, ontology, report / "oof_validation.json")
    if selected:
        b_records = read_jsonl(ROOT / plan["b_neural_predictions"])
        if {str(d["pmc_id"]) for d in b_records} != target_ids:
            raise ValueError("B neural predictions must cover the exact blind B set.")
        b_queries = sorted({span["text"] for record in b_records for span in eligible_spans(record, minimum)})
        b_retrieved = index.retrieve(b_queries)
        write_json(ROOT / "work/span_ner/semantic_linking/b_retrieval.json", b_retrieved)
        linker = SpanLinker(ontology, ROOT / plan["umls_path"]).fit(documents)
        b_neural = {str(d["pmc_id"]): d for d in b_records}
        b_base = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["baseline_b"])}
        predictions = []
        for document in targets:
            identifier = str(document["pmc_id"])
            candidates = proposals(document, b_neural[identifier], linker, b_retrieved, minimum)
            predictions.append(predict(document, b_base[identifier]["entities"], candidates, selected["configuration"], plan["margin_threshold"]))
        write_jsonl(output, predictions)
        summary["b_validation"] = validate_file(output, targets, ontology, report / "b_validation.json")
        summary["output_sha256"] = digest_file(output)
    summary["elapsed_seconds"] = time.time() - started
    write_json(report / "summary.json", summary)
    print(json.dumps({"selected": summary["selected"], "best_score": best["metrics"]["score"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/semantic_linking/plan.json")
    run(parser.parse_args().plan)
