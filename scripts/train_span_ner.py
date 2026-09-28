"""Train an overlapping phenotype span detector on fixed document-level folds.

Metrics are exact character-span diagnostics, not the competition's four scores.
HPO normalization and patient association are deliberately outside this model.
The CLI can show help without importing optional GPU-training dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TRAIN_SHA256 = "5029544d59a1e812de83c07a56d841c09b88cae222d4266fd2930b4c9a224786"
FIT_ALL_SCOPE = "Fit all 80 official labeled training documents with a fixed configured epoch budget; no validation metrics or best-checkpoint selection"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patientphex.data import read_jsonl
from patientphex.span_data import LABEL_NAMES, build_span_windows
from patientphex.span_losses import boundary_ranking_loss


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    import torch

    temporary = path.with_name(path.name + ".tmp")
    # Save tensor state dictionaries, never a pickled nn.Module instance.
    torch.save(value, temporary)
    os.replace(temporary, path)


def prepare_output(path: Path, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"Output directory already exists: {path}; choose a new run directory or --resume")
        if not (path / "last.pt").is_file() or not (path / "manifest.json").is_file():
            raise ValueError("--resume requires last.pt and manifest.json in the output directory")
    elif resume:
        raise FileNotFoundError(f"Cannot resume missing output directory: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def select_fold(documents: list[dict], split: dict, fold: int) -> tuple[list[dict], list[dict]]:
    folds = split["folds"]
    if not 0 <= fold < len(folds):
        raise ValueError(f"fold must be between 0 and {len(folds) - 1}")
    ids = [str(item) for group in folds for item in group]
    source_ids = [str(document["pmc_id"]) for document in documents]
    if len(ids) != len(set(ids)) or len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate document IDs in the data or saved folds")
    if set(ids) != set(source_ids):
        raise ValueError("Saved folds must partition exactly the training input documents")
    validation_ids = {str(item) for item in folds[fold]}
    training = [doc for doc in documents if str(doc["pmc_id"]) not in validation_ids]
    validation = [doc for doc in documents if str(doc["pmc_id"]) in validation_ids]
    if not training or not validation:
        raise ValueError("Both the training and validation document sets must be nonempty")
    return training, validation


def select_fit_all(documents: list[dict]) -> list[dict]:
    """Require the complete explicitly labeled training cohort, never A/B input."""
    if len(documents) != 80:
        raise ValueError("--fit-all requires exactly the 80 official labeled training documents")
    identifiers = [str(document["pmc_id"]) for document in documents]
    if len(set(identifiers)) != 80:
        raise ValueError("--fit-all requires 80 unique training document IDs")
    if any(not isinstance(document.get("entities"), list) or not document["entities"] for document in documents):
        raise ValueError("--fit-all requires explicit nonempty entity labels in every document; unlabeled A/B data is forbidden")
    return list(documents)


def load_training_documents(data_path: Path, split_path: Path, fold: int | None,
                            fit_all: bool = False) -> tuple[list[dict], list[dict], str | None]:
    documents = read_jsonl(data_path)
    if fit_all:
        training = select_fit_all(documents)
        if sha256_file(data_path) != OFFICIAL_TRAIN_SHA256:
            raise ValueError("--fit-all input fingerprint differs from the immutable official 80-document training JSONL")
        # The split file and any validation data are intentionally never opened.
        return training, [], None
    split = json.loads(split_path.read_text(encoding="utf-8"))
    training, validation = select_fold(documents, split, fold)
    return training, validation, sha256_file(split_path)


def valid_span_count(window: dict, width: int) -> int:
    total = run = 0
    for offset, visible, special in zip(
        window["offset_mapping"], window["attention_mask"], window["special_tokens_mask"],
    ):
        run = run + 1 if offset is not None and visible and not special else 0
        total += min(run, width)
    return total


def class_statistics(windows: list[dict], width: int, cap: float) -> dict:
    candidates = sum(valid_span_count(window, width) for window in windows)
    ignored = [sum(label == target for window in windows
                   for _, _, label in window.get("ignored_candidate_labels", []))
               for target in range(len(LABEL_NAMES))]
    supervised = [candidates - count for count in ignored]
    positives = [sum(span["labels"][label] for window in windows for span in window["span_labels"])
                 for label in range(len(LABEL_NAMES))]
    if not candidates or any(count == 0 for count in positives):
        raise ValueError("Training requires valid candidate spans and examples of both positive and NO labels")
    weights = [min(cap, max(1.0, (total - count) / count)) for total, count in zip(supervised, positives)]
    # Normalize the expected weighted loss magnitude separately for each label.
    normalizers = [(total - count + weight * count) / total
                   for total, count, weight in zip(supervised, positives, weights)]
    return {"candidate_spans": candidates, "positive_counts": positives,
            "supervised_candidates_per_label": supervised, "ignored_negative_candidates_per_label": ignored,
            "pos_weight": weights, "loss_normalizer": normalizers, "pos_weight_cap": cap}


def exclude_unencoded_gold(windows: list[dict], audit: dict, width: int) -> None:
    """Never train a dropped gold annotation as a negative candidate.

    For each unencodable entity, ignore intersecting negatives of that label.
    Keep exact positive targets, other label channels, and inference unchanged.
    Invalid source annotations require repair rather than guessing an exclusion.
    """
    by_document = {}
    for entity in audit["unencoded_entities"]:
        if any(reason in {"invalid_entity_span", "entity_not_in_passage", "text_mismatch"}
               for reason in entity["reasons"]):
            raise ValueError(f"Invalid source annotation requires repair before training: {entity}")
        by_document.setdefault(str(entity["pmc_id"]), []).append(entity)
    for window in windows:
        offsets = window["offset_mapping"]
        valid = [pair is not None and bool(visible) and not bool(special)
                 for pair, visible, special in zip(offsets, window["attention_mask"], window["special_tokens_mask"])]
        positives = {(span["start_token"], span["end_token"] - span["start_token"], label)
                     for span in window["span_labels"] for label, value in enumerate(span["labels"]) if value}
        ignored = set()
        for entity in by_document.get(str(window["pmc_id"]), []):
            left, right = entity["offset"], entity["offset"] + entity["length"]
            label = 1 if entity.get("note") == "NO" else 0
            content = [pair for pair in offsets if pair is not None]
            if not content or content[-1][1] <= left or content[0][0] >= right:
                continue
            for start in range(len(offsets)):
                for delta in range(min(width, len(offsets) - start)):
                    if not valid[start + delta]:
                        break
                    if offsets[start][0] < right and offsets[start + delta][1] > left:
                        key = start, delta, label
                        if key not in positives:
                            ignored.add(key)
        if ignored:
            window["ignored_candidate_labels"] = sorted(ignored)


def candidate_mask(valid_tokens: Any, width: int) -> Any:
    """Reject a candidate if ANY token inside it is padding or special."""
    import torch

    batch, length = valid_tokens.shape
    bad_prefix = torch.cat((torch.zeros(batch, 1, dtype=torch.long, device=valid_tokens.device),
                            (~valid_tokens).long().cumsum(dim=1)), dim=1)
    columns = []
    for delta in range(width):
        usable = max(0, length - delta)
        column = torch.zeros(batch, length, dtype=torch.bool, device=valid_tokens.device)
        if usable:
            column[:, :usable] = bad_prefix[:, delta + 1:delta + 1 + usable] == bad_prefix[:, :usable]
        columns.append(column)
    return torch.stack(columns, dim=2)


@dataclass
class SpanCollator:
    pad_token_id: int
    max_span_width: int
    include_labels: bool = True

    def __call__(self, windows: list[dict]) -> dict:
        import torch

        length = max(len(window["input_ids"]) for window in windows)
        shape = len(windows), length
        inputs = {"input_ids": torch.full(shape, self.pad_token_id, dtype=torch.long),
                  "attention_mask": torch.zeros(shape, dtype=torch.long)}
        token_types = ["token_type_ids" in window for window in windows]
        if any(token_types) and not all(token_types):
            raise ValueError("Inconsistent token_type_ids in tokenized windows")
        if all(token_types):
            inputs["token_type_ids"] = torch.zeros(shape, dtype=torch.long)
        valid = torch.zeros(shape, dtype=torch.bool)
        targets = torch.zeros((*shape, self.max_span_width, len(LABEL_NAMES))) if self.include_labels else None
        for row, window in enumerate(windows):
            size = len(window["input_ids"])
            for name in inputs:
                inputs[name][row, :size] = torch.tensor(window[name], dtype=torch.long)
            valid[row, :size] = torch.tensor([
                offset is not None and bool(visible) and not bool(special)
                for offset, visible, special in zip(window["offset_mapping"],
                                                    window["attention_mask"], window["special_tokens_mask"])
            ])
            if targets is not None:
                for span in window["span_labels"]:
                    start, end = span["start_token"], span["end_token"]
                    targets[row, start, end - start] = torch.tensor(span["labels"], dtype=torch.float32)
        mask = candidate_mask(valid, self.max_span_width)
        if targets is not None and bool((targets.bool() & ~mask.unsqueeze(-1)).any()):
            raise ValueError("A gold span includes special/padding tokens or exceeds the candidate mask")
        loss_mask = mask.unsqueeze(-1).expand(-1, -1, -1, len(LABEL_NAMES)).clone()
        if self.include_labels:
            for row, window in enumerate(windows):
                for start, delta, label in window.get("ignored_candidate_labels", []):
                    loss_mask[row, start, delta, label] = False
        return {"inputs": inputs, "mask": mask, "loss_mask": loss_mask, "targets": targets, "windows": windows}


def build_model(encoder: Any, max_span_width: int, rank: int = 64, dropout: float = 0.1) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class SpanModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.width = max_span_width
            self.rank = rank
            self.dropout = nn.Dropout(dropout)
            self.start_projection = nn.Linear(encoder.config.hidden_size, len(LABEL_NAMES) * rank)
            self.end_projection = nn.Linear(encoder.config.hidden_size, len(LABEL_NAMES) * rank)
            self.start_linear = nn.Linear(encoder.config.hidden_size, len(LABEL_NAMES))
            self.end_linear = nn.Linear(encoder.config.hidden_size, len(LABEL_NAMES))
            self.width_bias = nn.Parameter(torch.full((max_span_width, len(LABEL_NAMES)), -2.0))
            nn.init.zeros_(self.start_linear.bias)
            nn.init.zeros_(self.end_linear.bias)

        def forward(self, **inputs: Any) -> Any:
            hidden = self.dropout(self.encoder(**inputs).last_hidden_state)
            batch, length, _ = hidden.shape
            query = functional.gelu(self.start_projection(hidden)).view(batch, length, len(LABEL_NAMES), self.rank)
            key = functional.gelu(self.end_projection(hidden)).view(batch, length, len(LABEL_NAMES), self.rank)
            left, right = self.start_linear(hidden), self.end_linear(hidden)
            scores = []
            # Only one B x L x C x rank product at a time, never B x L x W x H.
            for delta in range(self.width):
                usable = max(0, length - delta)
                logits = (query[:, :usable] * key[:, delta:delta + usable]).sum(-1) / math.sqrt(self.rank)
                logits = logits + left[:, :usable] + right[:, delta:delta + usable] + self.width_bias[delta]
                scores.append(functional.pad(logits, (0, 0, 0, length - usable)))
            return torch.stack(scores, dim=2)

    return SpanModel()


def span_loss(logits: Any, targets: Any, mask: Any, pos_weight: Any, normalizer: Any) -> Any:
    from torch.nn import functional as functional

    losses = functional.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), pos_weight=pos_weight, reduction="none",
    ) / normalizer
    per_label = (losses * mask).sum(dim=(0, 1, 2)) / mask.sum(dim=(0, 1, 2)).clamp_min(1)
    return per_label.mean()


def gold_span_sets(documents: list[dict]) -> list[set]:
    gold = [set() for _ in LABEL_NAMES]
    for document in documents:
        for entity in document.get("entities", []):
            label = 1 if entity.get("note") == "NO" else 0
            gold[label].add((str(document["pmc_id"]), entity["offset"], entity["length"]))
    return gold


def score_sets(predicted: set, gold: set) -> dict:
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": true_positive, "fp": len(predicted - gold), "fn": len(gold - predicted),
            "predicted": len(predicted), "gold": len(gold)}


def score_predictions(predictions: dict, documents: list[dict]) -> dict:
    gold = gold_span_sets(documents)
    predicted = [{key[:3] for key in predictions if key[3] == label} for label in range(len(LABEL_NAMES))]
    metrics = {name: score_sets(predicted[label], gold[label]) for label, name in enumerate(LABEL_NAMES)}
    metrics["micro"] = score_sets(
        {(*key, label) for label, items in enumerate(predicted) for key in items},
        {(*key, label) for label, items in enumerate(gold) for key in items},
    )
    metrics["any_span"] = score_sets(set.union(*predicted), set.union(*gold))
    metrics["scope"] = "Exact original-character spans, positive/NO; all gold spans retained in recall; no HPO or patient scores"
    return metrics


def decode_batch(probabilities: Any, mask: Any, windows: list[dict], predictions: dict, threshold: float) -> None:
    hits = ((probabilities >= threshold) & mask.unsqueeze(-1)).nonzero(as_tuple=False).tolist()
    for row, start, delta, label in hits:
        offsets = windows[row]["offset_mapping"]
        left, right = offsets[start][0], offsets[start + delta][1]
        if right <= left:
            continue
        key = str(windows[row]["pmc_id"]), left, right - left, label
        score = float(probabilities[row, start, delta, label])
        predictions[key] = max(predictions.get(key, 0.0), score)


def write_predictions(path: Path, predictions: dict, documents: list[dict]) -> None:
    grouped = {str(document["pmc_id"]): {} for document in documents}
    sources = {str(document["pmc_id"]): document for document in documents}
    for (pmc_id, offset, length, label), score in sorted(predictions.items()):
        passage = next(paragraph for paragraph in sources[pmc_id]["full_text"]
                       if paragraph["offset"] <= offset and offset + length <= paragraph["offset"] + len(paragraph["text"]))
        text = passage["text"][offset - passage["offset"]:offset - passage["offset"] + length]
        span = grouped[pmc_id].setdefault((offset, length), {
            "offset": offset, "length": length, "text": text, "labels": [], "scores": {},
        })
        span["labels"].append(LABEL_NAMES[label])
        span["scores"][LABEL_NAMES[label]] = score
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for document in documents:
            stream.write(json.dumps({"pmc_id": document["pmc_id"], "spans": list(grouped[str(document["pmc_id"])].values())},
                                    ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def evaluate(model: Any, loader: Any, documents: list[dict], device: Any, fp16: bool, threshold: float) -> tuple[dict, dict]:
    import torch

    predictions = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = {name: value.to(device, non_blocking=True) for name, value in batch["inputs"].items()}
            # No labels, identifiers, or gold-derived candidates enter the model.
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                logits = model(**inputs)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Non-finite validation logits; refusing invalid checkpoint selection")
            probabilities = logits.float().sigmoid().cpu()
            decode_batch(probabilities, batch["mask"], batch["windows"], predictions, threshold)
    return score_predictions(predictions, documents), predictions


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-path", type=Path, required=True, help="Local Hugging Face encoder/tokenizer directory")
    result.add_argument("--model-source", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    result.add_argument("--model-revision", default="e1354b7a3a09615f6aba48dfad4b7a613eef7062")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--data", type=Path, default=ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl")
    result.add_argument("--split", type=Path, default=ROOT / "reports/cpu_baseline/split.json")
    result.add_argument("--fold", type=int, default=0)
    result.add_argument("--fit-all", action="store_true",
                        help="Train only the official labeled 80-document input with a fixed epoch budget; no validation or best checkpoint")
    result.add_argument("--epochs", type=int, default=5, help="Total epochs, including epochs restored with --resume")
    result.add_argument("--max-steps", type=int, default=0, help="Absolute optimizer-step limit; zero means all epochs")
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--eval-batch-size", type=int, default=8)
    result.add_argument("--gradient-accumulation", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=2e-5)
    result.add_argument("--head-learning-rate", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--warmup-ratio", type=float, default=0.1)
    result.add_argument("--gradient-clip", type=float, default=1.0)
    result.add_argument("--pos-weight-cap", type=float, default=1000.0)
    result.add_argument("--boundary-ranking-weight", type=float, default=0.0,
                        help="Training-only hardest same-start/end negative hinge weight; zero preserves the original objective")
    result.add_argument("--boundary-ranking-margin", type=float, default=1.0)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--max-length", type=int, default=384)
    result.add_argument("--stride", type=int, default=128)
    result.add_argument("--max-span-width", type=int, default=64)
    result.add_argument("--head-rank", type=int, default=64)
    result.add_argument("--dropout", type=float, default=0.1)
    result.add_argument("--threshold", type=float, default=0.5)
    result.add_argument("--device", default="cuda:0", help="CUDA failure is fatal; CPU must be explicitly requested")
    result.add_argument("--seed", type=int, default=20260928)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--eval-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--smoke", action="store_true", help="Cap training at 20 optimizer steps; still evaluate all validation documents")
    result.add_argument("--resume", action="store_true", help="Resume last.pt with identical data and schedule; never overwrite a fresh run")
    return result


def run(args: argparse.Namespace) -> dict:
    if args.fit_all:
        if args.smoke or args.max_steps:
            raise ValueError("--fit-all requires a complete fixed epoch schedule; --smoke and --max-steps are forbidden")
        if args.fold not in (0, None):
            raise ValueError("--fit-all cannot be combined with a nonzero --fold")
        args.fold = None
        args.eval_every_epoch = False
    if args.smoke:
        args.max_steps = min(args.max_steps or 20, 20)
    for name in ("epochs", "batch_size", "eval_batch_size", "gradient_accumulation", "max_length", "max_span_width", "head_rank", "threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_steps < 0 or args.num_workers < 0 or not 0 < args.threshold < 1:
        raise ValueError("Invalid step/worker count or threshold")
    if not (0 <= args.warmup_ratio < 1 and 0 <= args.dropout < 1 and args.pos_weight_cap >= 1):
        raise ValueError("Invalid warmup ratio, dropout, or class-weight cap")
    if any(not math.isfinite(value) or value < 0 for value in (args.boundary_ranking_weight, args.boundary_ranking_margin)):
        raise ValueError("Boundary-ranking weight and margin must be finite and nonnegative")
    if min(args.learning_rate, args.head_learning_rate, args.gradient_clip) <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rates and clipping must be positive; weight decay must be nonnegative")
    # Refuse an existing run before loading models or creating further artifacts.
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output directory already exists: {args.output_dir}; choose a new run directory or --resume")
    training_docs, validation_docs, split_hash = load_training_documents(args.data, args.split, args.fold, args.fit_all)
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {args.model_path}")

    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are explicit cpu or cuda[:index]")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing implicit CPU fallback")
        torch.cuda.set_device(device)
    fp16 = args.fp16 and device.type == "cuda"
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_output(args.output_dir, args.resume)
    started = time.time()

    def log(event: str, **values: Any) -> None:
        row = {"event": event, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "elapsed_seconds": round(time.time() - started, 3), **values}
        with (args.output_dir / "training.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define a padding token")
    training_windows, training_audit = build_span_windows(training_docs, tokenizer, args.max_length, args.stride, args.max_span_width)
    validation_audit = None
    validation_windows = []
    if not args.fit_all:
        _, validation_audit = build_span_windows(validation_docs, tokenizer, args.max_length, args.stride, args.max_span_width)
        blind_validation = [{"pmc_id": doc["pmc_id"], "full_text": doc["full_text"]} for doc in validation_docs]
        validation_windows, _ = build_span_windows(blind_validation, tokenizer, args.max_length, args.stride, args.max_span_width)
        validation_windows = [{key: value for key, value in window.items() if key != "span_labels"} for window in validation_windows]
    exclude_unencoded_gold(training_windows, training_audit, args.max_span_width)
    statistics = class_statistics(training_windows, args.max_span_width, args.pos_weight_cap)
    batches_per_epoch = math.ceil(len(training_windows) / args.batch_size)
    total_steps = math.ceil(batches_per_epoch / args.gradient_accumulation) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    model_files = {str(path.relative_to(args.model_path)): sha256_file(path)
                   for path in sorted(args.model_path.rglob("*"))
                   if path.is_file() and path.suffix in (".json", ".txt", ".bin", ".safetensors", ".model")}
    code_files = [Path(__file__), ROOT / "patientphex/span_data.py", ROOT / "patientphex/data.py",
                  ROOT / "patientphex/span_losses.py"]
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {
        "parameters": config, "fit_all": args.fit_all, "labels": list(LABEL_NAMES), "training_document_ids": [str(doc["pmc_id"]) for doc in training_docs],
        "validation_document_ids": [str(doc["pmc_id"]) for doc in validation_docs],
        "sha256": {"data": sha256_file(args.data), "split": split_hash,
                   "code": {str(path.relative_to(ROOT)): sha256_file(path) for path in code_files}, "model_files": model_files},
        "model": {"source": args.model_source, "revision": args.model_revision, "local_path": str(args.model_path.resolve()),
                  "provenance": "Source/revision supplied by CLI; local file hashes identify actual loaded artifacts"},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                    "cuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "fp16": fp16, "attention": "eager"},
        "training_windows": len(training_windows), "validation_windows": len(validation_windows),
        "batches_per_epoch": batches_per_epoch, "planned_optimizer_steps": total_steps, "class_statistics": statistics,
        "objective": {"base": "weighted_bce", "boundary_ranking_weight": args.boundary_ranking_weight,
                      "boundary_ranking_margin": args.boundary_ranking_margin,
                      "boundary_ranking_version": "same_endpoint_hardest_negative_v1",
                      "negative_policy": "Same start or end; exclude gold in any label and ignored candidates in any label"},
        "evaluation": FIT_ALL_SCOPE if args.fit_all else "Document-held-out exact span diagnostics only; validation used for checkpoint selection, not an independent test score",
    }
    if args.fit_all:
        manifest["fit_all_scope"] = FIT_ALL_SCOPE
        manifest["epoch_schedule"] = {"epochs": args.epochs, "selection": "Fixed before training; no validation-based early stopping or checkpoint selection"}
    if args.resume:
        previous = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
        excluded = {"resume", "output_dir", "num_workers", "threads", "device"}
        current_config = {key: value for key, value in config.items() if key not in excluded}
        previous_config = {key: value for key, value in previous["parameters"].items() if key not in excluded}
        if current_config != previous_config or previous["sha256"] != manifest["sha256"]:
            raise ValueError("Resume data/model/code hashes or training parameters differ from the original run")
        for key in ("torch", "transformers", "cuda", "fp16", "attention"):
            if previous["runtime"][key] != manifest["runtime"][key]:
                raise ValueError(f"Resume runtime differs for {key}; resume requires the original environment")
        if previous["runtime"]["device"].split(":")[0] != device.type:
            raise ValueError("Resume requires the original device type")
    else:
        atomic_json(args.output_dir / "manifest.json", manifest)
        atomic_json(args.output_dir / "alignment_audit.json", {"train": training_audit, "validation": validation_audit})
        tokenizer.save_pretrained(args.output_dir / "tokenizer")

    encoder = AutoModel.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False, attn_implementation="eager")
    if args.max_length > encoder.config.max_position_embeddings:
        raise ValueError("max_length exceeds the encoder's position embeddings")
    model = build_model(encoder, args.max_span_width, args.head_rank, args.dropout).to(device)
    if not args.resume:
        encoder.config.save_pretrained(args.output_dir / "encoder_config")
    groups = []
    for backbone in (True, False):
        for decay in (True, False):
            parameters = [parameter for name, parameter in model.named_parameters()
                          if parameter.requires_grad and name.startswith("encoder.") == backbone
                          and (parameter.ndim > 1) == decay]
            if parameters:
                groups.append({"params": parameters, "lr": args.learning_rate if backbone else args.head_learning_rate,
                               "weight_decay": args.weight_decay if decay else 0.0})
    optimizer = torch.optim.AdamW(groups)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    pos_weight = torch.tensor(statistics["pos_weight"], device=device)
    normalizer = torch.tensor(statistics["loss_normalizer"], device=device)
    validation_loader = None
    if not args.fit_all:
        validation_loader = DataLoader(validation_windows, batch_size=args.eval_batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=device.type == "cuda",
                                       generator=torch.Generator().manual_seed(args.seed),
                                       collate_fn=SpanCollator(tokenizer.pad_token_id, args.max_span_width, include_labels=False))
    start_epoch = next_batch = global_step = 0
    best_score = None if args.fit_all else -1.0
    best_epoch = None
    last_metrics = None
    if args.resume:
        checkpoint = torch.load(args.output_dir / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch, next_batch, global_step = checkpoint["epoch"], checkpoint["next_batch"], checkpoint["global_step"]
        best_score, best_epoch, last_metrics = checkpoint["best_score"], checkpoint["best_epoch"], checkpoint["metrics"]
        random.setstate(checkpoint["python_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        del checkpoint
        if global_step >= total_steps or start_epoch >= args.epochs:
            raise ValueError("This run already reached its configured training limit; use a new output directory")
    log("start", resume=args.resume, fit_all=args.fit_all, train_documents=len(training_docs), validation_documents=len(validation_docs),
        train_windows=len(training_windows), validation_windows=len(validation_windows), class_statistics=statistics,
        train_unencoded=training_audit["counts"]["unencoded_entities"],
        validation_unencoded=validation_audit["counts"]["unencoded_entities"] if validation_audit is not None else None,
        planned_steps=total_steps, device=str(device), fp16=fp16, runtime=manifest["runtime"], objective=manifest["objective"])
    start_summary = {"status": "running", "smoke": args.smoke, "optimizer_steps": global_step,
                     "metric_scope": manifest["evaluation"]}
    if args.fit_all:
        start_summary.update({"fit_all": True, "fit_all_scope": FIT_ALL_SCOPE, "metrics": None})
    else:
        start_summary.update({"best_epoch": best_epoch, "best_micro_f1": best_score})
    atomic_json(args.output_dir / "summary.json", start_summary)
    if args.resume and (args.output_dir / "failure.json").is_file():
        old_failure = json.loads((args.output_dir / "failure.json").read_text(encoding="utf-8"))
        atomic_json(args.output_dir / "failure.json", {**old_failure, "active": False, "recovered_by_resume": True})
    optimizer.zero_grad(set_to_none=True)
    stopped = False
    try:
        for epoch in range(start_epoch, args.epochs):
            loader = DataLoader(training_windows, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed + epoch),
                                collate_fn=SpanCollator(tokenizer.pad_token_id, args.max_span_width))
            model.train()
            group_loss = 0.0
            group_bce_loss = group_ranking_loss = 0.0
            skip_batches = next_batch if epoch == start_epoch else 0
            processed_batches = skip_batches
            for batch_index, batch in enumerate(loader):
                if batch_index < skip_batches:
                    continue
                group_start = batch_index - batch_index % args.gradient_accumulation
                group_size = min(args.gradient_accumulation, len(loader) - group_start)
                inputs = {key: value.to(device, non_blocking=True) for key, value in batch["inputs"].items()}
                targets, mask = batch["targets"].to(device), batch["loss_mask"].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                    logits = model(**inputs)
                    bce_loss = span_loss(logits, targets, mask, pos_weight, normalizer)
                    loss = bce_loss
                    ranking_loss = None
                    if args.boundary_ranking_weight > 0:
                        ranking_loss = boundary_ranking_loss(logits, targets, mask, args.boundary_ranking_margin)
                        loss = loss + args.boundary_ranking_weight * ranking_loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, batch {batch_index}")
                group_loss += float(loss.detach()) / group_size
                group_bce_loss += float(bce_loss.detach()) / group_size
                if ranking_loss is not None:
                    group_ranking_loss += float(ranking_loss.detach()) / group_size
                scaler.scale(loss / group_size).backward()
                processed_batches = batch_index + 1
                if processed_batches % args.gradient_accumulation and processed_batches != len(loader):
                    continue
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                if not fp16 and not bool(torch.isfinite(norm)):
                    raise FloatingPointError(f"Non-finite gradient at epoch {epoch + 1}, batch {batch_index}")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                applied = scaler.get_scale() >= previous_scale
                optimizer.zero_grad(set_to_none=True)
                if applied:
                    scheduler.step()
                    global_step += 1
                log("train_step", epoch=epoch + 1, batch=processed_batches, step=global_step, loss=group_loss,
                    span_bce_loss=group_bce_loss, boundary_ranking_loss=group_ranking_loss,
                    boundary_ranking_weight=args.boundary_ranking_weight,
                    gradient_norm=float(norm) if bool(torch.isfinite(norm)) else None, optimizer_step_applied=applied,
                    learning_rate=scheduler.get_last_lr()[0], amp_scale=scaler.get_scale(),
                    gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else None)
                group_loss = 0.0
                group_bce_loss = group_ranking_loss = 0.0
                if global_step >= total_steps:
                    stopped = True
                    break
            epoch_complete = processed_batches == len(loader)
            if not args.fit_all and (args.eval_every_epoch or stopped or epoch + 1 == args.epochs):
                last_metrics, predictions = evaluate(model, validation_loader, validation_docs, device, fp16, args.threshold)
                log("validation", epoch=epoch + 1, step=global_step, epoch_complete=epoch_complete, metrics=last_metrics)
                write_predictions(args.output_dir / "validation_predictions.jsonl", predictions, validation_docs)
                if last_metrics["micro"]["f1"] > best_score:
                    best_score, best_epoch = last_metrics["micro"]["f1"], epoch + 1
                    atomic_checkpoint(args.output_dir / "best.pt", {"model_state_dict": model.state_dict(),
                                      "epoch": epoch + 1, "global_step": global_step, "metrics": last_metrics,
                                      "model_parameters": {"max_span_width": args.max_span_width, "rank": args.head_rank, "dropout": args.dropout}})
                    write_predictions(args.output_dir / "best_validation_predictions.jsonl", predictions, validation_docs)
            next_epoch, next_batch = (epoch + 1, 0) if epoch_complete else (epoch, processed_batches)
            atomic_checkpoint(args.output_dir / "last.pt", {
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                "epoch": next_epoch, "next_batch": next_batch, "global_step": global_step,
                "best_score": best_score, "best_epoch": best_epoch, "metrics": last_metrics,
                "python_rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            })
            if args.fit_all and (stopped or epoch + 1 == args.epochs):
                atomic_checkpoint(args.output_dir / "final.pt", {
                    "model_state_dict": model.state_dict(), "epoch": epoch + 1,
                    "global_step": global_step, "metrics": None,
                    "model_parameters": {"max_span_width": args.max_span_width, "rank": args.head_rank, "dropout": args.dropout},
                })
            summary = {"status": "completed" if stopped or epoch + 1 == args.epochs else "running",
                       "smoke": args.smoke, "optimizer_steps": global_step, "epochs_completed": next_epoch,
                       "last_epoch_complete": epoch_complete, "best_epoch": best_epoch, "best_micro_f1": best_score,
                       "last_metrics": last_metrics, "duration_seconds": time.time() - started,
                       "training_documents": len(training_docs), "validation_documents": len(validation_docs),
                       "metric_scope": manifest["evaluation"], "threshold": args.threshold}
            if args.fit_all:
                summary.pop("best_epoch")
                summary.pop("best_micro_f1")
                summary.update({"fit_all": True, "fit_all_scope": FIT_ALL_SCOPE, "metrics": None,
                                "configured_epochs": args.epochs, "prediction_checkpoint": "last.pt"})
            atomic_json(args.output_dir / "summary.json", summary)
            if stopped:
                break
        log("complete", **summary)
        return summary
    except BaseException as error:
        log("failure", exception=type(error).__name__, message=str(error), step=global_step)
        failure = {"active": True, "exception": type(error).__name__, "message": str(error),
                   "step": global_step, "last_checkpoint_may_precede_failure": True}
        atomic_json(args.output_dir / "failure.json", failure)
        failed_summary = {"status": "failed", **failure}
        if args.fit_all:
            failed_summary.update({"fit_all": True, "fit_all_scope": FIT_ALL_SCOPE, "metrics": None})
        atomic_json(args.output_dir / "summary.json", failed_summary)
        raise


def main() -> None:
    args = parser().parse_args()
    existed = args.output_dir.exists()
    try:
        run(args)
    except BaseException as error:
        if not existed and args.output_dir.is_dir() and not (args.output_dir / "failure.json").exists():
            failure = {"active": True, "exception": type(error).__name__, "message": str(error),
                       "phase": "initialization", "resume_available": False}
            atomic_json(args.output_dir / "failure.json", failure)
            failed_summary = {"status": "failed", **failure}
            if args.fit_all:
                failed_summary.update({"fit_all": True, "fit_all_scope": FIT_ALL_SCOPE, "metrics": None})
            atomic_json(args.output_dir / "summary.json", failed_summary)
        raise


if __name__ == "__main__":
    main()
