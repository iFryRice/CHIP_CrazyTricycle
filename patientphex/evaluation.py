"""Transparent local approximation of the published PatientPheX metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


APPROXIMATIONS = [
    "No official scorer is supplied; these are local implementations of the published metrics.",
    "Main-text filtering excludes explicitly labelled reference/back-matter passages. "
    "Titles, abstracts, figures and tables remain eligible because they contain training labels.",
    "A gold -1 mention matches one prediction at its exact span regardless of predicted ID. "
    "Document-level -1 is one concept bucket per document; official handling is unspecified.",
    "Compound HPO strings are split for both entity and association scoring; unmapped "
    "association text is compared exactly, including case.",
    "Only NO denotes negation. Other gold notes, including D, remain positive. "
    "Gold IDs are evaluated as supplied without ontology repair.",
    "Zero denominators yield zero except a patient's two empty phenotype sets, "
    "whose precision, recall and F1 are all one as specified by the competition.",
]

NON_CONTENT_SECTIONS = frozenset(
    {
        "REF", "REFS", "REFERENCES", "REFERENCE", "BIBLIOGRAPHY", "ACK", "ACKNOWLEDGMENTS",
        "ACKNOWLEDGEMENTS", "FUNDING", "COMPETING_INTERESTS", "AUTHOR_CONTRIBUTIONS",
        "SUPPLEMENT", "SUPPLEMENTARY", "APPENDIX",
    }
)
NON_CONTENT_TYPES = frozenset({"ref", "reference", "references", "back", "footnote"})


def is_evaluation_passage(passage: dict[str, Any]) -> bool:
    """Keep content passages, including labelled title/abstract/figure examples."""
    section = str(passage.get("section_type", "")).upper().replace(" ", "_")
    passage_type = str(passage.get("type", "")).lower()
    return section not in NON_CONTENT_SECTIONS and passage_type not in NON_CONTENT_TYPES


def _index_documents(documents: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for document in documents:
        pmc_id = document["pmc_id"]
        if pmc_id in indexed:
            raise ValueError(f"Duplicate {label} document: {pmc_id}")
        indexed[pmc_id] = document
    return indexed


def _metrics(tp: int, fp: int, fn: int, *, empty_perfect: bool = False) -> dict[str, Any]:
    if empty_perfect and tp == fp == fn == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _positive_units(document: dict[str, Any], passages: list[dict[str, Any]]) -> set[tuple[int, int, str]]:
    units = set()
    for entity in document["entities"]:
        if entity.get("note") == "NO":
            continue
        offset, length = entity["offset"], entity["length"]
        containing = [
            passage for passage in passages
            if passage["offset"] <= offset
            and offset + length <= passage["offset"] + len(passage["text"])
        ]
        # Uncontained predictions are errors, rather than silently ignored.
        if containing and not any(is_evaluation_passage(passage) for passage in containing):
            continue
        units.update((offset, length, identifier.strip()) for identifier in entity["identifier"].split(";"))
    return units


def _mention_counts(gold: set[tuple[int, int, str]], predicted: set[tuple[int, int, str]]) -> tuple[int, int, int]:
    mapped_gold = {unit for unit in gold if unit[2] != "-1"}
    unmapped_gold = gold - mapped_gold
    matched = mapped_gold & predicted
    remaining = predicted - matched
    tp = len(matched)
    for offset, length, _ in sorted(unmapped_gold):
        candidates = sorted(unit for unit in remaining if unit[:2] == (offset, length))
        if candidates:
            # Prefer the literal -1 prediction when several IDs share this span.
            chosen = next((unit for unit in candidates if unit[2] == "-1"), candidates[0])
            remaining.remove(chosen)
            tp += 1
    return tp, len(predicted) - tp, len(gold) - tp


def _association_set(values: list[str]) -> set[str]:
    result = set()
    for value in values:
        # Raw unmapped text can contain punctuation; split HPO compounds only.
        if value.startswith("HP:"):
            result.update(part.strip() for part in value.split(";"))
        else:
            result.add(value)
    return result


def _patients(document: dict[str, Any], expected: set[str], label: str) -> dict[str, set[str]]:
    associations = {}
    for association in document["association"]:
        patient_id = association["patient_id"]
        if patient_id in associations:
            raise ValueError(f"Duplicate {label} patient {patient_id} in {document['pmc_id']}")
        associations[patient_id] = _association_set(association["phenotype"])
    if set(associations) != expected:
        raise ValueError(
            f"{label} patient coverage mismatch for {document['pmc_id']}: "
            f"missing={sorted(expected - set(associations))}, "
            f"unexpected={sorted(set(associations) - expected)}"
        )
    return associations


def evaluate(
    gold_documents: Iterable[dict[str, Any]],
    prediction_documents: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Score exact document/patient coverage using the four published F1 metrics."""
    gold_by_id = _index_documents(gold_documents, "gold")
    pred_by_id = _index_documents(prediction_documents, "prediction")
    if not gold_by_id:
        raise ValueError("Cannot evaluate an empty dataset")
    if set(gold_by_id) != set(pred_by_id):
        raise ValueError(
            f"Document coverage mismatch: missing={sorted(set(gold_by_id) - set(pred_by_id))}, "
            f"unexpected={sorted(set(pred_by_id) - set(gold_by_id))}"
        )
    totals = {name: [0, 0, 0] for name in ("mention", "document", "association_micro")}
    patient_metrics = []
    for pmc_id, gold in gold_by_id.items():
        predicted = pred_by_id[pmc_id]
        if predicted.get("pmid") != gold.get("pmid"):
            raise ValueError(f"PMID mismatch for {pmc_id}")
        passages = gold.get("full_text", [])
        gold_units = _positive_units(gold, passages)
        pred_units = _positive_units(predicted, passages)
        mention = _mention_counts(gold_units, pred_units)
        gold_concepts, pred_concepts = {unit[2] for unit in gold_units}, {unit[2] for unit in pred_units}
        document = (
            len(gold_concepts & pred_concepts), len(pred_concepts - gold_concepts), len(gold_concepts - pred_concepts)
        )
        for name, counts in (("mention", mention), ("document", document)):
            totals[name] = [previous + count for previous, count in zip(totals[name], counts)]
        expected = {patient["patient_id"] for patient in gold["patient"]}
        if len(expected) != len(gold["patient"]):
            raise ValueError(f"Duplicate source patient in {pmc_id}")
        gold_patients = _patients(gold, expected, "gold")
        pred_patients = _patients(predicted, expected, "prediction")
        for patient_id in sorted(expected):
            actual, proposed = gold_patients[patient_id], pred_patients[patient_id]
            counts = len(actual & proposed), len(proposed - actual), len(actual - proposed)
            totals["association_micro"] = [
                previous + count for previous, count in zip(totals["association_micro"], counts)
            ]
            patient_metrics.append(_metrics(*counts, empty_perfect=True))
    result = {name: _metrics(*counts) for name, counts in totals.items()}
    result["association_macro"] = {
        key: sum(patient[key] for patient in patient_metrics) / len(patient_metrics) if patient_metrics else 0.0
        for key in ("precision", "recall", "f1")
    }
    result["association_macro"]["patients"] = len(patient_metrics)
    result["score"] = sum(result[name]["f1"] for name in (
        "mention", "document", "association_micro", "association_macro"
    )) / 4
    result["counts"] = {"documents": len(gold_by_id), "patients": len(patient_metrics)}
    result["approximations"] = list(APPROXIMATIONS)
    return result
