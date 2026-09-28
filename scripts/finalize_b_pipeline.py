"""Fit the selected linking pipeline on official training labels and write B."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patientphex.data import digest_file, prediction_record, read_jsonl, write_json, write_jsonl
from patientphex.ontology import Ontology
from patientphex.span_linking import SpanLinker
from patientphex.validation import validate_submission
from scripts.evaluate_linked_pipeline import associations, build_entity_variants, fit_patient_model
from scripts.predict_span_ner import blind_documents
from scripts.train_span_ner import sha256_file


def validate_final_model(manifest: dict, summary: dict, inference: dict, selection: dict) -> None:
    """Bind completed final training and inference to the frozen development choice."""
    parameters = manifest["parameters"]
    epochs = parameters["epochs"]
    if (summary.get("status") != "completed" or not summary.get("fit_all")
            or summary.get("epochs_completed") != epochs or not summary.get("last_epoch_complete")):
        raise ValueError("Final all-document training has not completed its frozen epoch budget")
    if summary.get("last_metrics") is not None or summary.get("best_micro_f1") is not None:
        raise ValueError("An all-document final fit must not select a checkpoint using validation labels")
    if Path(inference["checkpoint_path"]).name != "last.pt" or inference["checkpoint_epoch"] != epochs:
        raise ValueError("B inference must use the completed last.pt checkpoint at the frozen final epoch")
    expected = selection["ner_training_parameters"]
    for key, value in expected.items():
        if parameters.get(key) != value:
            raise ValueError(f"Final NER parameter differs from the frozen selection: {key}")
    if manifest["sha256"]["model_files"] != selection["model_files_sha256"]:
        raise ValueError("Final encoder/tokenizer differs from the selected development model")


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    if args.report_dir.exists() or args.model_dir.exists() or args.output.exists():
        raise FileExistsError("Final B output, report and fitted-model directories must be new")
    task = json.loads(args.task.read_text(encoding="utf-8"))
    if task["phase"] != "B":
        raise ValueError("This entrypoint only accepts the declared B competition target")
    training_path, target_path = ROOT / task["training"]["path"], ROOT / task["inference_target"]["path"]
    for role, path in (("training", training_path), ("inference_target", target_path)):
        if digest_file(path) != task[role]["sha256"]:
            raise ValueError(f"Declared {role} data hash mismatch")
    documents, targets = read_jsonl(training_path), read_jsonl(target_path)
    train_ids, target_ids = {str(d["pmc_id"]) for d in documents}, {str(d["pmc_id"]) for d in targets}
    if train_ids & target_ids or any(d.get("entities") or d.get("association") for d in targets):
        raise ValueError("B input is not a disjoint blind target")
    if len(targets) != task["inference_target"]["documents"] or sum(len(d["patient"]) for d in targets) != task["inference_target"]["patients"]:
        raise ValueError("B document/patient coverage differs from the declared target")
    inference_path = args.spans.parent / "manifest.json"
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    model_manifest = json.loads((args.ner_run / "manifest.json").read_text(encoding="utf-8"))
    if not model_manifest["parameters"].get("fit_all") or model_manifest["validation_document_ids"]:
        raise ValueError("Final B requires a declared all-training-document fit without validation selection")
    if set(model_manifest["training_document_ids"]) != train_ids:
        raise ValueError("Final detector was not fitted on exactly the official training documents")
    if inference["source_manifest_sha256"] != digest_file(args.ner_run / "manifest.json"):
        raise ValueError("B spans do not belong to the supplied final detector")
    if inference["data_sha256"] != digest_file(target_path) or inference["prediction_sha256"] != digest_file(args.spans):
        raise ValueError("B inference data or predictions have changed")
    if inference["role"] != "target" or inference["gold_metrics_computed"] or not inference["answers_removed_before_windowing"]:
        raise ValueError("B inference role is invalid")
    if sha256_file(Path(inference["checkpoint_path"])) != inference["checkpoint_sha256"]:
        raise ValueError("Final detector checkpoint no longer matches B inference provenance")
    span_records = read_jsonl(args.spans)
    spans_by_id = {str(record["pmc_id"]): record for record in span_records}
    if set(spans_by_id) != target_ids:
        raise ValueError("B span predictions do not cover exactly B")
    comparison = json.loads(args.selection.read_text(encoding="utf-8"))
    if comparison.get("schema_version") != 1 or comparison.get("phase") != "B":
        raise ValueError("Require the frozen, provenance-bound B pipeline selection")
    summary = json.loads((args.ner_run / "summary.json").read_text(encoding="utf-8"))
    validate_final_model(model_manifest, summary, inference, comparison)
    if args.negative_weight != comparison["association_parameters"]["negative_weight"]:
        raise ValueError("Association negative weighting differs from development selection")
    resources = {"training_data": training_path, "ontology": ROOT / task["ontology"]["path"],
                 "umls_terms": args.umls_terms,
                 "umls_manifest": args.umls_terms.with_name(args.umls_terms.name + ".manifest.json")}
    for name, path in resources.items():
        if digest_file(path) != comparison["resource_sha256"][name]:
            raise ValueError(f"Final resource differs from development selection: {name}")
    for name, expected in comparison["inference_code_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"Final linking implementation changed after selection: {name}")
    for record in comparison["selection_sources"].values():
        if digest_file(record["path"]) != record["sha256"]:
            raise ValueError(f"Development selection source changed: {record['path']}")
    selected = comparison["selected"]
    span_threshold = float(selected["span_threshold"])
    if span_threshold < inference["score_floor"]:
        raise ValueError("Saved B score floor is higher than the selected threshold")
    ontology = Ontology(ROOT / task["ontology"]["path"])
    if ontology.version != task["ontology"]["version"]:
        raise ValueError("The final pipeline must use the competition HPO release")
    fixed = SpanLinker(ontology, args.umls_terms).fit([])
    trained = SpanLinker(ontology, args.umls_terms).fit(documents, forbidden_document_ids=target_ids)
    patient_model, _, provenance = fit_patient_model(documents, fixed, forbidden_document_ids=target_ids,
                                                    negative_weight=args.negative_weight)
    unmapped_texts = {str(entity["text"]) for doc in documents for entity in doc["entities"]
                      if "-1" in str(entity["identifier"]).split(";")}
    predictions, audits = [], []
    for document in blind_documents(targets):
        variants, audit = build_entity_variants(document, spans_by_id[str(document["pmc_id"])], fixed,
                                               trained, span_threshold, unmapped_texts)
        entities = variants[selected["entity_strategy"]]
        scores = patient_model.predict_scores(document, entities)
        linked = associations(document, scores, float(selected["association_threshold"]))
        predictions.append(prediction_record(document, entities, linked))
        audits.append({"pmc_id": document["pmc_id"], "entities": len(entities), "linking": audit})
    args.report_dir.mkdir(parents=True, exist_ok=False)
    args.model_dir.mkdir(parents=True, exist_ok=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report_dir / "validated_b_candidate.jsonl"
    write_jsonl(temporary, predictions)
    validation = validate_submission(temporary, targets, ontology)
    if not validation["document_order_preserved"]:
        raise ValueError("Final B record order changed")
    os.replace(temporary, args.output)
    validation["path"] = str(args.output.resolve())
    with (args.model_dir / "patient_linking.pkl").open("wb") as stream:
        pickle.dump(patient_model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(args.report_dir / "submission_validation.json", validation)
    write_json(args.report_dir / "linking_audit.json", audits)
    write_json(args.report_dir / "association_training.json", patient_model.training_diagnostics_)
    result = {
        "status": "completed", "phase": "B", "training_documents": len(documents),
        "target_documents": len(targets), "target_patients": sum(len(d["patient"]) for d in targets),
        "selected_configuration": selected, "selection_path": str(args.selection.resolve()),
        "selection_sha256": digest_file(args.selection), "target_sha256": digest_file(target_path),
        "training_sha256": digest_file(training_path), "umls_terms_sha256": digest_file(args.umls_terms),
        "ner_manifest_sha256": digest_file(args.ner_run / "manifest.json"),
        "inference_manifest_sha256": digest_file(inference_path),
        "source_code_sha256": {name: digest_file(ROOT / name) for name in (
            "scripts/finalize_b_pipeline.py", "scripts/evaluate_linked_pipeline.py",
            "patientphex/span_linking.py", "patientphex/patient_linking.py")},
        "association_candidate_provenance": provenance, "negative_weight": args.negative_weight,
        "submission": validation, "duration_seconds": time.time() - started,
        "scope": "Validated B prediction artifact. B has no local gold; no B score or improvement claim is made. No competition upload performed.",
    }
    write_json(args.report_dir / "manifest.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=ROOT / "experiments/span_ner/task.json")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--ner-run", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--umls-terms", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "submissions/patientphex_b.jsonl")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/b_supervised")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "work/span_ner/b_linking")
    parser.add_argument("--negative-weight", type=float, default=1.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
