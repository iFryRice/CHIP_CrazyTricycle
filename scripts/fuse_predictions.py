"""Fuse complete association predictions with one global deterministic rule."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.data import digest_file, read_jsonl
from patientphex.ontology import Ontology
from patientphex.validation import validate_submission


METHODS = (
    "cpu_union_gpu", "cpu_intersection_gpu", "cpu_union_rag",
    "cpu_intersection_rag", "majority", "cpu_plus_consensus",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def association_values(values):
    """Split HPO compounds only; preserve unmapped text exactly, including case."""
    require(isinstance(values, list), "Phenotypes must be a list.")
    result = []
    for value in values:
        require(isinstance(value, str) and bool(value.strip()) and value != "-1",
                "An association needs an HPO ID or exact unmapped text, never literal -1.")
        if value.startswith("HP:"):
            parts = value.split(";")
            require(all(re.fullmatch(r"HP:\d{7}", part) for part in parts),
                    "Invalid HPO association or compound identifier.")
        else:
            parts = [value]
        for part in parts:
            require(part not in result, "Duplicate association concept.")
            result.append(part)
    return result


def model_names(method):
    require(method in METHODS, f"Unknown fusion method: {method}")
    if method.endswith("_gpu"):
        return ("gpu",)
    if method.endswith("_rag"):
        return ("rag",)
    return ("gpu", "rag")


def index_documents(rows, label):
    result = {row["pmc_id"]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate {label} document IDs.")
    return result


def fuse(cpu, sources, method, *, gpu=None, rag=None, multi_patient_only=False):
    """Read only source PMID/patient metadata; never inspect source gold labels."""
    names = model_names(method)
    baseline = index_documents(cpu, "CPU")
    source = index_documents(sources, "source")
    require(bool(baseline) and set(baseline) <= set(source), "CPU/source coverage mismatch.")
    models = {"cpu": baseline}
    for name, rows in (("gpu", gpu), ("rag", rag)):
        if name in names:
            require(rows is not None, f"{name} predictions are required for {method}.")
            models[name] = index_documents(rows, name)
            require(set(models[name]) == set(baseline), f"Incomplete {name} document coverage.")
    output = copy.deepcopy(cpu)
    for record in output:
        identifier = record["pmc_id"]
        expected = [p["patient_id"] for p in source[identifier]["patient"]]
        require(len(expected) == len(set(expected)), "Duplicate source patient IDs.")
        require(record["pmid"] == source[identifier]["pmid"], "CPU PMID differs from source.")
        associations = {}
        for name, documents in models.items():
            row = documents[identifier]
            require(row["pmid"] == record["pmid"] and row["entities"] == record["entities"],
                    f"{name} modified PMID/entities for {identifier}.")
            indexed = {a["patient_id"]: association_values(a["phenotype"])
                       for a in row["association"]}
            require(len(indexed) == len(row["association"]) and set(indexed) == set(expected),
                    f"{name} patient coverage mismatch for {identifier}.")
            associations[name] = indexed
        if multi_patient_only and len(expected) <= 1:
            continue
        record["association"] = []
        for patient_id in expected:
            values = {name: set(rows[patient_id]) for name, rows in associations.items()}
            c, g, r = values["cpu"], values.get("gpu", set()), values.get("rag", set())
            if method == "cpu_union_gpu":
                selected = c | g
            elif method == "cpu_intersection_gpu":
                selected = c & g
            elif method == "cpu_union_rag":
                selected = c | r
            elif method == "cpu_intersection_rag":
                selected = c & r
            elif method == "majority":
                selected = (c & g) | (c & r) | (g & r)
            else:
                selected = c | (g & r)
            # Keep CPU ordering, then append new concepts in model output order.
            ordered = list(dict.fromkeys(value for name in ("cpu", *names)
                                         for value in associations[name][patient_id]))
            record["association"].append({"patient_id": patient_id,
                                           "phenotype": [v for v in ordered if v in selected]})
    require(all(a["entities"] == b["entities"] and a["pmid"] == b["pmid"]
                and a["pmc_id"] == b["pmc_id"] for a, b in zip(cpu, output)),
            "Fusion changed CPU entities, PMID, or document order.")
    return output


def complete_predictions(path, summary_path, cpu_path, expected_ids):
    require(summary_path is not None, f"A completed inference summary is required for {path}.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    count = len(expected_ids)
    require(summary.get("all_documents_refined") is True
            and summary.get("documents") == count and summary.get("gpu_documents") == count
            and summary.get("cpu_documents") == 0, "Inference contains pending or fallback CPU records.")
    statuses = summary.get("status", [])
    require(len(statuses) == count and {s.get("pmc_id") for s in statuses} == expected_ids
            and all(s.get("source") == "gpu" for s in statuses), "Incomplete per-document GPU status.")
    require(summary.get("input_sha256") == digest_file(cpu_path), "Model input differs from CPU snapshot.")
    require(Path(summary.get("output", "")).resolve() == path.resolve(), "Model summary/output path mismatch.")
    return read_jsonl(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--rag", type=Path)
    parser.add_argument("--gpu-summary", type=Path)
    parser.add_argument("--rag-summary", type=Path)
    parser.add_argument("--data", type=Path, required=True, help="Source dataset for patient IDs and validation.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--multi-patient-only", action="store_true",
                        help="Keep all CPU associations in documents with at most one patient.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protected = {p.resolve() for p in (args.cpu, args.gpu, args.rag, args.data,
                                       args.gpu_summary, args.rag_summary, Path(__file__)) if p}
    require(args.output.resolve() not in protected, "Output must not overwrite an input or this script.")
    require(not args.output.exists(), "Output already exists; choose an intentional method-specific new path.")
    cpu = read_jsonl(args.cpu)
    ids = {row["pmc_id"] for row in cpu}
    sources = [d for d in read_jsonl(args.data) if d["pmc_id"] in ids]
    ontology = Ontology(args.data.parent / "hp.obo")
    validate_submission(args.cpu, sources, ontology)
    models = {}
    for name in model_names(args.method):
        path, summary = getattr(args, name), getattr(args, name + "_summary")
        require(path is not None, f"--{name} is required for {args.method}.")
        models[name] = complete_predictions(path, summary, args.cpu, ids)
        validate_submission(path, sources, ontology)
    output = fuse(cpu, sources, args.method, multi_patient_only=args.multi_patient_only, **models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                                   for row in output), encoding="utf-8")
    report = validate_submission(args.output, sources, ontology)
    print(json.dumps({"method": args.method, "multi_patient_only": args.multi_patient_only,
                      "validation": report,
                      "entities_unchanged": True, "source_gold_labels_used": False},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
