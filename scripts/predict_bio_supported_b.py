"""Apply a confirmed, fully fitted BIO acronym guard to the blind B set."""

import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.bio_corroboration import revise_entities
from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.ontology import Ontology
from patientphex.span_data import build_span_windows
from scripts.optimize_cpu_association import blind, from_scores, validate_file
from scripts.train_bio_ner import BioCollator, build_model, predict_spans


def require_final_bio(manifest, summary, checkpoint, training_ids, plan_sha256):
    ids = manifest.get("training_document_ids", [])
    if not manifest.get("fit_all") or manifest.get("smoke") or manifest.get("validation_document_ids"):
        raise ValueError("Final BIO model must be a full-training, non-smoke fit without validation labels.")
    if len(ids) != 80 or len(set(ids)) != 80 or set(ids) != set(training_ids):
        raise ValueError("Final BIO model must use exactly the 80 official training documents.")
    if summary.get("status") != "completed" or summary.get("epochs_completed") != 5 or summary.get("validation_documents") != 0:
        raise ValueError("Final BIO training did not complete the fixed epoch budget.")
    if checkpoint.get("epoch") != 5 or checkpoint.get("completed_epoch") is not True or checkpoint.get("ner_family") != "bio":
        raise ValueError("Final BIO checkpoint is incomplete or incompatible.")
    if any(value.get("plan_sha256") != plan_sha256 for value in (manifest, summary, checkpoint)):
        raise ValueError("Final BIO training plan identity changed.")


def run():
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    started = time.time()
    plan_path = ROOT / "experiments/bio_corroboration/finalization_plan.json"
    plan = json.loads(plan_path.read_text())
    for filename, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if digest_file(ROOT / filename) != expected:
            raise ValueError(f"Finalization input/code changed: {filename}")
    confirmation = json.loads((ROOT / "reports/bio_corroboration/confirmation/summary.json").read_text())
    if confirmation.get("promoted") is not True or not all(confirmation["checks"].values()):
        raise ValueError("The BIO policy did not pass full confirmation.")
    work, output = ROOT / "work/span_ner/bio_b_support", ROOT / "submissions/patientphex_b_supported.jsonl"
    if work.exists() or output.exists():
        raise FileExistsError("Refusing to overwrite a prior B inference run.")
    directory = ROOT / "work/span_ner/bio-final-all80"
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if digest_file(directory / "last.pt") != summary["checkpoint_sha256"]:
        raise ValueError("Final BIO checkpoint bytes changed.")
    checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    training = read_jsonl(ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl")
    require_final_bio(manifest, summary, checkpoint, [str(d["pmc_id"]) for d in training], digest_file(ROOT / "experiments/bio_ner/plan.json"))
    documents = read_jsonl(ROOT / "PatientPheX-V1-B/PatientPheX-V1-B.jsonl")
    if len(documents) != 100 or sum(len(d["patient"]) for d in documents) != 244 or any(d.get("entities") or d.get("association") for d in documents):
        raise ValueError("B must contain all 100 blind documents and 244 patients.")
    if set(str(d["pmc_id"]) for d in documents) & set(manifest["training_document_ids"]):
        raise ValueError("B overlaps the training documents.")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared CUDA inference route is unavailable.")
    model_path = ROOT / "models/biomedbert"
    for filename, expected in manifest["base_model_files"].items():
        if digest_file(model_path / filename) != expected:
            raise ValueError("Base model/tokenizer changed.")
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True, trust_remote_code=False)
    model = build_model(AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, attn_implementation="eager"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    device = torch.device("cuda:0")
    model.to(device)
    parameters = manifest["parameters"]
    blinded = [blind(document) for document in documents]
    windows, _ = build_span_windows(blinded, tokenizer, parameters["max_length"], parameters["stride"], 64)
    windows = [{key: value for key, value in window.items() if key != "span_labels"} for window in windows]
    loader = DataLoader(windows, batch_size=parameters["batch_size"], shuffle=False, num_workers=0, pin_memory=True, collate_fn=BioCollator(tokenizer, False))
    raw = predict_spans(model, loader, blinded, device)
    raw_index = {str(record["pmc_id"]): record["spans"] for record in raw}
    if set(raw_index) != {str(d["pmc_id"]) for d in documents}:
        raise ValueError("B BIO inference did not cover every target.")
    work.mkdir(parents=True)
    write_jsonl(work / "spans.jsonl", raw)
    with (ROOT / "work/span_ner/association_context/model.pkl").open("rb") as handle:
        association = pickle.load(handle)
    baseline = read_jsonl(ROOT / "submissions/patientphex_b_context.jsonl")
    base_index = {str(d["pmc_id"]): d for d in baseline}
    reproduced = [from_scores(d, base_index[str(d["pmc_id"])]["entities"],
                    association.predict_scores(blind(d), base_index[str(d["pmc_id"])]["entities"]), 0.5) for d in documents]
    if reproduced != baseline:
        raise ValueError("The frozen patient model no longer reproduces B.")
    predictions, edits = [], {}
    for document in documents:
        identifier = str(document["pmc_id"])
        entities, edits[identifier] = revise_entities(base_index[identifier]["entities"], raw_index[identifier], [], veto_acronyms=True)
        predictions.append(from_scores(document, entities, association.predict_scores(blind(document), entities), 0.5))
    write_jsonl(output, predictions)
    report = ROOT / "reports/bio_corroboration/final"
    report.mkdir()
    validation = validate_file(output, documents, Ontology(ROOT / "PatientPheX-V1-A/hp.obo"), report / "b_validation.json")
    write_json(report / "b_edits.json", edits)
    result = {"validation": validation, "role": "target", "answers_removed_before_windowing": True, "gold_metrics_computed": False,
              "checkpoint_sha256": summary["checkpoint_sha256"], "checkpoint_epoch": 5, "training_documents": 80,
              "raw_spans_sha256": digest_file(work / "spans.jsonl"), "source_training_manifest_sha256": digest_file(directory / "manifest.json"),
              "source_training_summary_sha256": digest_file(directory / "summary.json"), "finalization_plan_sha256": digest_file(plan_path),
              "confirmation_summary_sha256": digest_file(ROOT / "reports/bio_corroboration/confirmation/summary.json"),
              "elapsed_seconds": time.time() - started}
    write_json(report / "summary.json", result)
    write_json(work / "manifest.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run()
