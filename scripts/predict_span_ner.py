"""Infer original-character spans from a saved detector without reading answers."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patientphex.data import read_jsonl
from patientphex.span_data import build_span_windows
from scripts.train_span_ner import (
    SpanCollator, atomic_json, build_model, decode_batch, sha256_file, write_predictions,
)


def blind_documents(documents: list[dict]) -> list[dict]:
    """Never expose entity or association answers to the inference window builder."""
    return [{**document, "entities": [], "association": []} for document in documents]


def select_documents(documents: list[dict], manifest: dict, subset: str, role: str) -> list[dict]:
    training_ids = set(manifest["training_document_ids"])
    validation_ids = set(manifest.get("validation_document_ids", []))
    if subset != "all":
        wanted = training_ids if subset == "training" else validation_ids
        documents = [document for document in documents if str(document["pmc_id"]) in wanted]
        if {str(document["pmc_id"]) for document in documents} != wanted:
            raise ValueError("Input does not contain every document in the requested checkpoint subset")
    if not documents:
        raise ValueError("No inference documents selected")
    if role == "target":
        if any(document.get("entities") or document.get("association") for document in documents):
            raise ValueError("Final target input must not contain answer labels")
        if (training_ids | validation_ids).intersection(str(document["pmc_id"]) for document in documents):
            raise ValueError("Target inference documents overlap checkpoint training or model-selection documents")
    if role == "validation" and training_ids.intersection(str(document["pmc_id"]) for document in documents):
        raise ValueError("Held-out inference documents overlap checkpoint training documents")
    return blind_documents(documents)


def run(args: argparse.Namespace) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    if not 0 < args.score_floor < 1:
        raise ValueError("score-floor must be in (0, 1)")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace inference artifacts: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA inference, but CUDA is unavailable")
    torch.set_num_threads(args.threads)
    torch.manual_seed(20260928)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    source_manifest = args.run_dir / "manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    parameters = manifest["parameters"]
    checkpoint_path = args.run_dir / args.checkpoint
    documents = select_documents(read_jsonl(args.data), manifest, args.subset, args.role)
    model_path = args.model_path or Path(parameters["model_path"])
    for name, expected in manifest["sha256"]["model_files"].items():
        if sha256_file(model_path / name) != expected:
            raise ValueError(f"Encoder/tokenizer file does not match the trained model: {name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    windows, _ = build_span_windows(documents, tokenizer, max_length=parameters["max_length"],
                                    stride=parameters["stride"], max_span_width=parameters["max_span_width"])
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                       attn_implementation="eager")
    model = build_model(encoder, parameters["max_span_width"], parameters["head_rank"], parameters["dropout"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    checkpoint_epoch = checkpoint["epoch"]
    del checkpoint
    device = torch.device(args.device)
    model.to(device).eval()
    fp16 = bool(parameters["fp16"] and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loader = DataLoader(windows, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=SpanCollator(tokenizer.pad_token_id, parameters["max_span_width"], include_labels=False))
    predictions = {}
    started = time.time()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            inputs = {name: value.to(device) for name, value in batch["inputs"].items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                logits = model(**inputs)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Non-finite inference logits")
            decode_batch(logits.float().sigmoid().cpu(), batch["mask"], batch["windows"], predictions, args.score_floor)
            if (index + 1) % 50 == 0:
                print(json.dumps({"event": "inference", "batches": index + 1, "total_batches": len(loader)}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_predictions(args.output_dir / "spans.jsonl", predictions, documents)
    result = {
        "status": "completed", "role": args.role, "subset": args.subset,
        "answers_removed_before_windowing": True, "gold_metrics_computed": False,
        "documents": len(documents), "document_ids": [str(document["pmc_id"]) for document in documents],
        "windows": len(windows), "span_label_predictions": len(predictions), "score_floor": args.score_floor,
        "duration_seconds": time.time() - started, "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_path": str(checkpoint_path.resolve()), "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_manifest_sha256": sha256_file(source_manifest), "data_sha256": sha256_file(args.data),
        "prediction_sha256": sha256_file(args.output_dir / "spans.jsonl"),
        "code_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/predict_span_ner.py", "scripts/train_span_ner.py", "patientphex/span_data.py")},
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else None,
        "scope": "Raw span candidates only, not a competition submission or an evaluation score",
    }
    atomic_json(args.output_dir / "manifest.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("best.pt", "last.pt"), default="best.pt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--role", choices=("validation", "training_candidates", "target"), required=True)
    parser.add_argument("--subset", choices=("all", "training", "validation"), default="all")
    parser.add_argument("--score-floor", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
