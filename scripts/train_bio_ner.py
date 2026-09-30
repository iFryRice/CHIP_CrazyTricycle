"""Train a fixed-epoch BiomedBERT BIO detector with auditable flat supervision."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.bio_data import LABELS, add_window_probabilities, bio_targets, decode_probabilities
from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.span_data import build_span_windows
from scripts.train_span_ner import atomic_checkpoint, load_training_documents


def build_model(encoder):
    import torch.nn as nn

    class BioModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(encoder.config.hidden_size, 3)

        def forward(self, **inputs):
            return self.classifier(self.dropout(self.encoder(**inputs).last_hidden_state))

    return BioModel()


class BioCollator:
    def __init__(self, tokenizer, include_labels):
        self.tokenizer = tokenizer
        self.include_labels = include_labels

    def __call__(self, windows):
        import torch

        inputs = self.tokenizer.pad([{key: window[key] for key in ("input_ids", "attention_mask", "token_type_ids") if key in window}
                                     for window in windows], padding=True, return_tensors="pt")
        output = {"inputs": inputs, "windows": windows}
        if self.include_labels:
            length = inputs["input_ids"].shape[1]
            output["labels"] = torch.tensor([window["bio_labels"] + [-100] * (length - len(window["bio_labels"])) for window in windows], dtype=torch.long)
        return output


def predict_spans(model, loader, documents, device):
    import torch

    model.eval()
    accumulated = {}
    with torch.inference_mode():
        for batch in loader:
            inputs = {name: value.to(device) for name, value in batch["inputs"].items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**inputs)
            probabilities = logits.float().softmax(dim=-1).cpu().numpy()
            for window, values in zip(batch["windows"], probabilities, strict=True):
                add_window_probabilities(accumulated, window, values)
    return decode_probabilities(documents, accumulated)


def exact_span_metrics(documents, predictions):
    indexed = {str(d["pmc_id"]): d for d in predictions}
    tp = fp = fn = 0
    for document in documents:
        gold = {(e["offset"], e["length"]) for e in document["entities"]}
        proposed = {(s["offset"], s["length"]) for s in indexed[str(document["pmc_id"])]["spans"]}
        tp += len(gold & proposed)
        fp += len(proposed - gold)
        fn += len(gold - proposed)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            "scope": "Exact spans only, including negated and nested gold; no HPO or patient score."}


def run(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    for path, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if digest_file(ROOT / path) != expected:
            raise ValueError(f"Frozen input/code mismatch: {path}")
    if args.output_dir.exists():
        raise FileExistsError("Refusing to overwrite a previous BIO run.")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared CUDA training route is unavailable.")
    parameters = plan["parameters"]
    random.seed(parameters["seed"])
    torch.manual_seed(parameters["seed"])
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    training, validation, split_hash = load_training_documents(ROOT / plan["train_path"], ROOT / plan["split_path"], args.fold, args.fit_all)
    reference = json.loads((ROOT / plan["reference_ner_manifest"]).read_text())
    model_path = ROOT / plan["model_path"]
    for name, expected in reference["sha256"]["model_files"].items():
        if digest_file(model_path / name) != expected:
            raise ValueError(f"Base model bytes changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True, trust_remote_code=False)
    windows, alignment = build_span_windows(training, tokenizer, parameters["max_length"], parameters["stride"], 64)
    training_index = {str(d["pmc_id"]): d for d in training}
    selected, omitted = set(), set()
    token_counts = Counter()
    for window in windows:
        labels, audit = bio_targets(window, training_index[str(window["pmc_id"])])
        window["bio_labels"] = labels
        token_counts.update(labels)
        selected.update((str(window["pmc_id"]), offset, length) for offset, length in audit["selected"])
        omitted.update((str(window["pmc_id"]), offset, length) for offset, length in audit["omitted_overlaps"])
    windows = [window for window in windows if any(label >= 0 for label in window["bio_labels"])]
    blind_validation = [{**d, "entities": [], "association": []} for d in validation]
    validation_windows, _ = build_span_windows(blind_validation, tokenizer, parameters["max_length"], parameters["stride"], 64) if validation else ([], {})
    validation_windows = [{k: v for k, v in window.items() if k != "span_labels"} for window in validation_windows]
    if not token_counts[1] or not token_counts[2]:
        raise ValueError("BIO training supervision is missing B or I labels.")
    args.output_dir.mkdir(parents=True)
    manifest = {"ner_family": "bio", "plan_sha256": digest_file(args.plan), "parameters": parameters,
                "training_document_ids": sorted(training_index), "validation_document_ids": sorted(str(d["pmc_id"]) for d in validation),
                "fit_all": args.fit_all, "smoke": args.max_steps is not None,
                "labels": list(LABELS), "split_sha256": split_hash, "train_sha256": digest_file(ROOT / plan["train_path"]),
                "code_sha256": plan["code_sha256"], "base_model_files": reference["sha256"]["model_files"],
                "training_windows": len(windows), "validation_windows": len(validation_windows),
                "projection": {"selected_unique_spans": len(selected), "omitted_overlap_unique_spans": len(omitted),
                               "omitted": sorted(omitted), "token_counts": dict(token_counts),
                               "policy": "Longest nonoverlapping gold spans for BIO training; clipped or unencoded gold tokens ignored; all original gold retained for evaluation."},
                "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "fp16": True},
                "selection": "Fixed five epochs; no validation-based checkpoint selection or early stopping."}
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "alignment_audit.json", alignment)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, attn_implementation="eager")
    model = build_model(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning_rate"], weight_decay=0.01)
    generator = torch.Generator().manual_seed(parameters["seed"])
    loader = DataLoader(windows, batch_size=parameters["batch_size"], shuffle=True, generator=generator,
                        num_workers=0, pin_memory=True, collate_fn=BioCollator(tokenizer, True))
    validation_loader = DataLoader(validation_windows, batch_size=parameters["batch_size"], shuffle=False,
                                   num_workers=0, pin_memory=True, collate_fn=BioCollator(tokenizer, False))
    total_steps = len(loader) * parameters["epochs"]
    warmup = max(1, int(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda", init_scale=parameters["amp_initial_scale"])
    torch.cuda.reset_peak_memory_stats()
    log_path = args.output_dir / "training.jsonl"

    def emit(event, **fields):
        row = {"event": event, "time_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.time() - started, **fields}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    emit("start", training_documents=len(training), validation_documents=len(validation), training_windows=len(windows),
         planned_steps=total_steps, projection=manifest["projection"] | {"omitted": None})
    step = applied = skipped = epochs_completed = 0
    last_metrics = None
    for epoch in range(1, parameters["epochs"] + 1):
        model.train()
        loss_total = 0.0
        completed_epoch = True
        for batch_index, batch in enumerate(loader):
            inputs = {name: value.to(device) for name, value in batch["inputs"].items()}
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**inputs)
                loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite BIO training loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            update_applied = scaler.get_scale() >= old_scale
            if update_applied:
                scheduler.step()
                applied += 1
            else:
                skipped += 1
            step += 1
            loss_total += float(loss)
            if step <= 5 or step % 25 == 0:
                emit("train_step", epoch=epoch, step=step, planned_steps=total_steps, loss=float(loss),
                     gradient_norm=float(gradient_norm) if math.isfinite(float(gradient_norm)) else None, optimizer_step_applied=update_applied,
                     amp_scale_before=old_scale, amp_scale_after=scaler.get_scale(),
                     learning_rate=scheduler.get_last_lr()[0], peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
            if args.max_steps is not None and step >= args.max_steps:
                completed_epoch = batch_index + 1 == len(loader)
                break
        if completed_epoch:
            epochs_completed = epoch
        atomic_checkpoint(args.output_dir / "last.pt", {"model_state_dict": model.state_dict(), "epoch": epoch,
                          "completed_epoch": completed_epoch, "plan_sha256": digest_file(args.plan), "ner_family": "bio"})
        if validation and completed_epoch and args.max_steps is None:
            predictions = predict_spans(model, validation_loader, blind_validation, device)
            last_metrics = exact_span_metrics(validation, predictions)
            write_jsonl(args.output_dir / "last_validation_spans.jsonl", predictions)
            emit("epoch_complete", epoch=epoch, loss=loss_total / len(loader), span_metrics=last_metrics)
        if args.max_steps is not None and step >= args.max_steps:
            break
    summary = {"status": "smoke_complete" if args.max_steps else "completed", "ner_family": "bio", "epochs_completed": epochs_completed,
               "configured_epochs": parameters["epochs"], "optimizer_steps": applied, "amp_skips": skipped,
               "training_documents": len(training), "validation_documents": len(validation), "last_metrics": last_metrics,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3, "elapsed_seconds": time.time() - started,
               "checkpoint_sha256": digest_file(args.output_dir / "last.pt"), "plan_sha256": digest_file(args.plan)}
    if not args.max_steps and epochs_completed != parameters["epochs"]:
        raise RuntimeError("BIO run did not complete its fixed epoch budget.")
    if skipped > max(3, step * 0.05):
        raise RuntimeError("Excessive AMP overflow; inspect training before trusting this run.")
    write_json(args.output_dir / "summary.json", summary)
    emit("complete", **summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/bio_ner/plan.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", type=int)
    group.add_argument("--fit-all", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
