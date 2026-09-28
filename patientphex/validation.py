"""Strict UTF-8 JSONL submission checks without changing submitted predictions."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


MAX_SUBMISSION_BYTES = 100_000_000
HPO_ID = re.compile(r"HP:\d{7}\Z")
DOCUMENT_KEYS = {"pmc_id", "pmid", "entities", "association"}
ENTITY_KEYS = {"identifier", "type", "offset", "length", "text", "note"}
ASSOCIATION_KEYS = {"patient_id", "phenotype"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON number: {value}")


def _identifiers(value: Any, allowed: set[str], *, unmapped: bool, context: str) -> list[str]:
    _require(isinstance(value, str) and bool(value.strip()), f"{context}: identifier must be a nonblank string")
    if value == "-1" and unmapped:
        return [value]
    parts = value.split(";")
    _require(len(parts) == len(set(parts)), f"{context}: duplicate compound identifier")
    for part in parts:
        if part == "-1" and unmapped:
            continue
        _require(bool(HPO_ID.fullmatch(part)) and part in allowed, f"{context}: invalid/nonbranch HPO identifier {part!r}")
    return parts


def validate_submission(
    path: str | Path,
    source_documents: Iterable[dict[str, Any]],
    ontology: Any,
) -> dict[str, Any]:
    """Validate a completed target-set file; raise ValueError on any invalid record.

    The ontology object must expose ``allowed_ids`` containing valid HPO branch IDs.
    Empty patient phenotype lists are legal. Individual empty documents are reported;
    a submission with no entity or association predictions is rejected as unfinished.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Cannot access submission file: {path}") from exc
    _require(0 < size < MAX_SUBMISSION_BYTES, f"Submission must be nonempty and smaller than {MAX_SUBMISSION_BYTES} bytes")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read submission file: {path}") from exc
    _require(not payload.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM is not permitted")
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Submission must use valid UTF-8") from exc
    sources = {}
    for document in source_documents:
        pmc_id = document["pmc_id"]
        _require(pmc_id not in sources, f"Duplicate source document {pmc_id}")
        sources[pmc_id] = document
    _require(bool(sources), "Source dataset must not be empty")
    allowed = set(ontology.allowed_ids)
    _require(bool(allowed), "Ontology contains no allowed HPO identifiers")
    records = []
    lines = content.split("\n")
    if lines[-1] == "":
        lines.pop()
    for line_number, line in enumerate(lines, 1):
        _require(bool(line.strip()), f"Line {line_number}: empty JSONL record")
        try:
            record = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Line {line_number}: invalid JSON: {exc}") from exc
        _require(isinstance(record, dict), f"Line {line_number}: record must be an object")
        records.append(record)
    _require(bool(records), "Submission contains no records")
    seen_documents = set()
    total_entities = total_patients = total_associations = negated = unmapped = compounds = 0
    empty_documents = []
    warnings = []
    for line_number, document in enumerate(records, 1):
        context = f"Line {line_number}"
        _require(set(document) == DOCUMENT_KEYS, f"{context}: expected root keys {sorted(DOCUMENT_KEYS)}")
        pmc_id = document["pmc_id"]
        _require(isinstance(pmc_id, str) and bool(pmc_id), f"{context}: pmc_id must be a nonblank string")
        _require(pmc_id in sources, f"{context}: unexpected document {pmc_id}")
        _require(pmc_id not in seen_documents, f"{context}: duplicate document {pmc_id}")
        seen_documents.add(pmc_id)
        source = sources[pmc_id]
        _require(document["pmid"] == source["pmid"], f"{context}: PMID differs from source")
        _require(isinstance(document["entities"], list), f"{context}: entities must be a list")
        _require(isinstance(document["association"], list), f"{context}: association must be a list")
        seen_entities = set()
        raw_phenotypes = set()
        for index, entity in enumerate(document["entities"]):
            entity_context = f"{pmc_id} entity {index}"
            _require(isinstance(entity, dict) and set(entity) == ENTITY_KEYS, f"{entity_context}: invalid entity keys")
            identifiers = _identifiers(entity["identifier"], allowed, unmapped=True, context=entity_context)
            _require(entity["type"] == "Phenotype", f"{entity_context}: type must be Phenotype")
            offset, length, text = entity["offset"], entity["length"], entity["text"]
            _require(type(offset) is int and offset >= 0, f"{entity_context}: offset must be a nonnegative integer")
            _require(type(length) is int and length > 0, f"{entity_context}: length must be a positive integer")
            _require(isinstance(text, str) and bool(text.strip()), f"{entity_context}: text must be nonblank")
            _require(length == len(text), f"{entity_context}: length differs from Python Unicode text length")
            _require(entity["note"] in (None, "NO"), f"{entity_context}: note must be null or NO")
            _require(any(
                passage["offset"] <= offset
                and offset + length <= passage["offset"] + len(passage["text"])
                and passage["text"][offset - passage["offset"]:offset - passage["offset"] + length] == text
                for passage in source["full_text"]
            ), f"{entity_context}: entity span/text does not exactly match a source paragraph")
            # Duplicating a unit inside different compound representations is also invalid.
            for identifier in identifiers:
                unit = offset, length, identifier
                _require(unit not in seen_entities, f"{entity_context}: duplicate entity unit")
                seen_entities.add(unit)
            if "-1" in identifiers and entity["note"] is None:
                raw_phenotypes.add(text)
            total_entities += 1
            negated += entity["note"] == "NO"
            unmapped += "-1" in identifiers
            compounds += len(identifiers) > 1
        expected_patients = {patient["patient_id"] for patient in source["patient"]}
        _require(len(expected_patients) == len(source["patient"]), f"{pmc_id}: duplicate patient in source")
        seen_patients = set()
        for index, association in enumerate(document["association"]):
            association_context = f"{pmc_id} association {index}"
            _require(isinstance(association, dict) and set(association) == ASSOCIATION_KEYS,
                     f"{association_context}: invalid association keys")
            patient_id = association["patient_id"]
            _require(isinstance(patient_id, str) and patient_id in expected_patients,
                     f"{association_context}: unexpected patient {patient_id!r}")
            _require(patient_id not in seen_patients, f"{association_context}: duplicate patient")
            seen_patients.add(patient_id)
            values = association["phenotype"]
            _require(isinstance(values, list), f"{association_context}: phenotype must be a list")
            seen_concepts = set()
            for value in values:
                _require(isinstance(value, str) and bool(value.strip()), f"{association_context}: phenotype must be nonblank")
                if value.startswith("HP:"):
                    concepts = _identifiers(value, allowed, unmapped=False, context=association_context)
                else:
                    # Task 2 may recover a phenotype missed by task 1. Permit exact
                    # source text rather than forcing the two outputs to be identical.
                    _require(value != "-1" and (
                        value in raw_phenotypes or any(value in passage["text"] for passage in source["full_text"])
                    ), f"{association_context}: unmapped phenotype is not source text")
                    concepts = [value]
                for concept in concepts:
                    _require(concept not in seen_concepts, f"{association_context}: duplicate phenotype {concept!r}")
                    seen_concepts.add(concept)
            total_associations += len(seen_concepts)
        _require(seen_patients == expected_patients,
                 f"{pmc_id}: missing patients {sorted(expected_patients - seen_patients)}")
        total_patients += len(seen_patients)
        if not document["entities"]:
            empty_documents.append(pmc_id)
    _require(seen_documents == set(sources), f"Missing documents {sorted(set(sources) - seen_documents)}")
    _require(total_entities + total_associations > 0, "Submission is an all-empty prediction placeholder")
    if empty_documents:
        warnings.append(f"Documents with no predicted entities: {', '.join(empty_documents)}")
    order_preserved = [record["pmc_id"] for record in records] == list(sources)
    if not order_preserved:
        warnings.append("Document order differs from the source dataset")
    return {
        "valid": True,
        "path": str(path.resolve()),
        "bytes": size,
        "limit_bytes_exclusive": MAX_SUBMISSION_BYTES,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "documents": len(records),
        "patients": total_patients,
        "entities": total_entities,
        "association_pairs": total_associations,
        "negated_entities": negated,
        "unmapped_entities": unmapped,
        "compound_entities": compounds,
        "document_order_preserved": order_preserved,
        "empty_entity_documents": empty_documents,
        "warnings": warnings,
    }
