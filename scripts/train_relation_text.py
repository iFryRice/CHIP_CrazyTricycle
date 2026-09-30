"""Train a fixed-budget BiomedBERT association classifier without held-out labels."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.ontology import Ontology
from patientphex.patient_linking import _check_provenance
from patientphex.relation_text import Collator, build_model, configure_tokenizer, encode, examples, supervise
from scripts.optimize_cpu_association import blind
from scripts.run_llm_association import load_plan
from scripts.train_span_ner import atomic_checkpoint


def run(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer
    import transformers

    started = time.time()
    plan = load_plan(args.plan)
    output = ROOT / plan["work_dir"] / ("smoke" if args.smoke else f"fold{args.fold}")
    if output.exists():
        raise FileExistsError("Preserve earlier supervised relation runs.")
    if not torch.cuda.is_available() or torch.__version__ != "2.6.0+cu124" or transformers.__version__ != "4.49.0":
        raise RuntimeError("The accepted root CUDA environment changed.")
    if args.fold not in plan["pilot_folds"]:
        raise ValueError("Fold is outside the declared training budget.")
    parameters = plan["parameters"]
    random.seed(parameters["seed"])
    torch.manual_seed(parameters["seed"])
    torch.set_num_threads(2)
    documents = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["train_path"])}
    folds = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    if sorted(sum(folds, [])) != sorted(documents):
        raise ValueError("Training partition coverage changed.")
    validation_ids = set(folds[args.fold])
    training_ids = set(documents)-validation_ids
    training = [documents[i] for i in sorted(training_ids)]
    validation = [blind(documents[i]) for i in folds[args.fold]]
    candidate_dir = ROOT / plan["candidate_root"] / f"fold{args.fold}"
    provenance = json.loads((candidate_dir / "candidate_provenance.json").read_text())
    _check_provenance(provenance, training_ids, validation_ids)
    candidates = {str(d["pmc_id"]): d["entities"] for d in read_jsonl(candidate_dir / "training_candidates.jsonl")}
    if set(candidates) != training_ids:
        raise ValueError("Training candidate coverage differs from the permitted fold.")
    baseline = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["baseline_oof"])}
    if set(baseline) != set(documents):
        raise ValueError("Baseline coverage differs from the source corpus.")
    reference = json.loads((ROOT / plan["reference_ner_manifest"]).read_text())
    model_path = ROOT / plan["model_path"]
    for name, expected in reference["sha256"]["model_files"].items():
        if digest_file(model_path / name) != expected:
            raise ValueError(f"Fixed pretrained encoder changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    configure_tokenizer(tokenizer)
    ontology = Ontology(ROOT / plan["ontology_path"])
    raw_training = [row for document in training
                    for row in examples(blind(document), candidates[str(document["pmc_id"])], ontology)]
    raw_training = supervise(raw_training, training, training_ids, validation_ids)
    raw_validation = [row for document in validation
                      for row in examples(document, baseline[str(document["pmc_id"])]["entities"], ontology)]
    train_rows = [encode(tokenizer, row) for row in raw_training]
    val_rows = [encode(tokenizer, row) for row in raw_validation]
    if {row["label"] for row in train_rows} != {0, 1}:
        raise ValueError("Association training requires both label classes.")
    if len({row["task_id"] for row in train_rows+val_rows}) != len(train_rows)+len(val_rows):
        raise ValueError("Duplicate or overlapping training/validation examples.")
    if any("label" in row or "weight" in row for row in val_rows):
        raise ValueError("Held-out labels reached model input construction.")
    output.mkdir(parents=True)
    write_json(output / "training_examples.json", raw_training)
    write_json(output / "validation_examples.json", raw_validation)
    manifest = {"plan_sha256": digest_file(args.plan), "fold": args.fold, "smoke": args.smoke,
                "training_document_ids": sorted(training_ids), "validation_document_ids": folds[args.fold],
                "candidate_provenance_sha256": digest_file(candidate_dir / "candidate_provenance.json"),
                "training_examples_sha256": digest_file(output / "training_examples.json"),
                "validation_examples_sha256": digest_file(output / "validation_examples.json"),
                "training_pairs": len(train_rows), "validation_pairs": len(val_rows),
                "training_label_counts": dict(Counter(row["label"] for row in train_rows)),
                "parameters": parameters, "model_files": reference["sha256"]["model_files"],
                "max_input_tokens": max(len(row["input_ids"]) for row in train_rows+val_rows),
                "cropped_examples": sum(row["cropped_context_tokens"] > 0 for row in train_rows+val_rows),
                "cropped_context_tokens": sum(row["cropped_context_tokens"] for row in train_rows+val_rows),
                "input_policy": "Three nearest positive occurrences, target-reference context, explicit target/other/entity markers; fixed per-fragment token budgets preserve focus spans. Original strings and offsets remain in example records.",
                "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
                "selection": "Fixed final epoch; no validation labels, checkpoint selection or early stopping in training."}
    write_json(output / "manifest.json", manifest)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                        attn_implementation="eager")
    encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model = build_model(encoder).cuda()
    initial_head = model.classifier.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning_rate"], weight_decay=0.01)
    loader = DataLoader(train_rows, batch_size=parameters["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(parameters["seed"]),
                        num_workers=0, collate_fn=Collator(tokenizer))
    total_steps = len(loader)*parameters["epochs"]
    warmup = max(1, int(total_steps*0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: s/warmup if s < warmup else max(0.0, (total_steps-s)/max(1, total_steps-warmup)))
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    step = applied = skipped = epochs_completed = 0
    torch.cuda.reset_peak_memory_stats()

    def emit(event, **fields):
        row = {"event": event, "time_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.time()-started, **fields}
        with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)

    emit("start", training_pairs=len(train_rows), validation_pairs=len(val_rows), planned_steps=total_steps,
         smoke=args.smoke, physical_gpu=manifest["physical_gpu"])
    for epoch in range(1, parameters["epochs"]+1):
        model.train()
        loss_total = 0.0
        for batch in loader:
            inputs = {name: value.cuda() for name, value in batch["inputs"].items()}
            labels, weights = batch["labels"].cuda(), batch["weights"].cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**inputs)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")*weights).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite relation training loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            accepted = scaler.get_scale() >= old_scale
            if accepted:
                applied += 1
                scheduler.step()
            else:
                skipped += 1
            step += 1
            loss_total += float(loss)
            if step <= 5 or step % 25 == 0:
                emit("train_step", epoch=epoch, step=step, planned_steps=total_steps, loss=float(loss),
                     gradient_norm=float(norm) if math.isfinite(float(norm)) else None,
                     optimizer_step_applied=accepted, amp_scale=scaler.get_scale())
            if args.smoke and step >= plan["smoke_steps"]:
                break
        if args.smoke:
            break
        epochs_completed = epoch
        emit("epoch_complete", epoch=epoch, mean_loss=loss_total/len(loader))
    if not applied or torch.equal(initial_head, model.classifier.weight) or skipped > max(3, step*0.05):
        raise RuntimeError("Optimizer update or numerical stability witness failed.")
    if not args.smoke and epochs_completed != parameters["epochs"]:
        raise RuntimeError("Fixed epoch budget did not complete.")
    atomic_checkpoint(output / "last.pt", {"model_state_dict": model.state_dict(), "plan_sha256": digest_file(args.plan),
                      "fold": args.fold, "epochs_completed": epochs_completed, "smoke": args.smoke})
    model.eval()
    predicted = []
    inference_rows = val_rows[:parameters["batch_size"]] if args.smoke else val_rows
    with torch.inference_mode():
        for batch in DataLoader(inference_rows, batch_size=parameters["batch_size"], shuffle=False, collate_fn=Collator(tokenizer)):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                probabilities = model(**{name: value.cuda() for name, value in batch["inputs"].items()}).float().sigmoid()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("Nonfinite held-out probabilities.")
            for row, probability in zip(batch["rows"], probabilities.cpu().tolist(), strict=True):
                predicted.append({**{key: row[key] for key in ["task_id", "pmc_id", "patient_id", "concept"]}, "score": probability})
    write_json(output / "validation_scores.json", predicted)
    summary = {"status": "smoke_complete" if args.smoke else "completed", "plan_sha256": digest_file(args.plan),
               "epochs_completed": epochs_completed, "optimizer_steps": applied, "amp_skips": skipped,
               "training_pairs": len(train_rows), "validation_pairs": len(predicted),
               "checkpoint_sha256": digest_file(output / "last.pt"),
               "scores_sha256": digest_file(output / "validation_scores.json"),
               "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
               "elapsed_seconds": time.time()-started}
    write_json(output / "summary.json", summary)
    emit("complete", **summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
