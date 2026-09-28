"""Deterministic JSONL input/output and document-level splitting."""

import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    identifiers = [str(record["pmc_id"]) for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate document identifiers in {path}")
    return records


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document_folds(documents, folds=5, seed=20260927):
    """Stratify by single/multiple patients without inspecting gold labels."""
    if folds < 2 or folds > len(documents):
        raise ValueError("Invalid fold count")
    buckets = {False: [], True: []}
    for document in documents:
        buckets[len(document["patient"]) > 1].append(document)
    result = [[] for _ in range(folds)]
    cursor = 0
    for key in (False, True):
        ordered = sorted(
            buckets[key],
            key=lambda document: hashlib.sha256(
                f"{seed}:{document['pmc_id']}".encode()
            ).hexdigest(),
        )
        for document in ordered:
            result[cursor % folds].append(document)
            cursor += 1
    return [sorted(fold, key=lambda document: str(document["pmc_id"])) for fold in result]


def prediction_record(document, entities, associations):
    return {
        "pmc_id": document["pmc_id"],
        "pmid": document["pmid"],
        "entities": entities,
        "association": associations,
    }
