"""Summarize complete GPU results on all 80 and the 72 non-pilot documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_results import ROOT, build_report, comparison
from patientphex.data import digest_file, read_jsonl


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_complete_summary(summary_path, candidate_path, baseline_path, baseline_ids):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("all_documents_refined") is True,
            "Inference is not complete; pending/fallback CPU records must not be scored as GPU results.")
    require(summary.get("documents") == 80 and summary.get("gpu_documents") == 80
            and summary.get("cpu_documents") == 0, "Expected 80 GPU and zero CPU records.")
    require(summary.get("entities_unchanged") is True, "Summary does not confirm unchanged CPU entities.")
    statuses = summary.get("status", [])
    require(len(statuses) == 80 and {s.get("pmc_id") for s in statuses} == baseline_ids
            and all(s.get("source") == "gpu" for s in statuses),
            "Status records must uniquely cover all 80 documents with source=gpu.")
    require(summary.get("input_sha256") == digest_file(baseline_path),
            "Inference input hash differs from the fixed CPU OOF baseline.")
    require(Path(summary.get("output", "")).resolve() == candidate_path.resolve(),
            "Candidate path differs from the inference summary output path.")
    return summary


def score_scope(ids, gold, baseline, candidate, per_document):
    result = comparison([gold[i] for i in ids], [baseline[i] for i in ids],
                        [candidate[i] for i in ids])
    rows = [per_document[i] for i in ids]
    epsilon = 1e-12
    result["document_outcomes"] = {
        "documents": len(ids),
        "criterion": "Per-document total score; +/- 1e-12 tolerance.",
        "improved": sum(r["delta"]["score"] > epsilon for r in rows),
        "worsened": sum(r["delta"]["score"] < -epsilon for r in rows),
        "score_unchanged": sum(abs(r["delta"]["score"]) <= epsilon for r in rows),
        "association_output_changed": sum(r["association_changed"] for r in rows),
    }
    result["groups"] = {}
    for name, predicate in (("single_patient", lambda n: n == 1),
                            ("multiple_patients", lambda n: n > 1)):
        members = [i for i in ids if predicate(len(gold[i]["patient"]))]
        if not members:
            result["groups"][name] = {"documents": 0}
            continue
        group = comparison([gold[i] for i in members], [baseline[i] for i in members],
                           [candidate[i] for i in members])
        group["document_outcomes"] = {
            "improved": sum(per_document[i]["delta"]["score"] > epsilon for i in members),
            "worsened": sum(per_document[i]["delta"]["score"] < -epsilon for i in members),
            "score_unchanged": sum(abs(per_document[i]["delta"]["score"]) <= epsilon for i in members),
        }
        result["groups"][name] = group
    result["per_document"] = rows
    return result


def summarize(args):
    baseline_rows = read_jsonl(args.baseline)
    baseline_ids = {d["pmc_id"] for d in baseline_rows}
    require(len(baseline_rows) == 80, "The fixed baseline must contain exactly 80 OOF documents.")
    summary = load_complete_summary(args.summary, args.candidate, args.baseline, baseline_ids)
    candidate_hash = digest_file(args.candidate)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    pilot = [d["pmc_id"] for d in selection["documents"]]
    require(len(pilot) == 8 and len(set(pilot)) == 8 and set(pilot) <= baseline_ids,
            "Pilot selection must contain eight distinct baseline documents.")
    full = build_report(args.gold, args.baseline, args.candidate)
    require(full["valid"], f"Strict comparison validation failed: {full['errors']}")
    require(full["scope"]["is_complete_80_document_training_set"],
            "Comparison does not cover the complete labelled 80-document corpus.")
    require(full["files"]["candidate"]["sha256"] == candidate_hash,
            "Candidate changed while validation was running.")
    gold = {d["pmc_id"]: d for d in read_jsonl(args.gold)}
    baseline = {d["pmc_id"]: d for d in baseline_rows}
    candidate = {d["pmc_id"]: d for d in read_jsonl(args.candidate)}
    per_document = {d["pmc_id"]: d for d in full["per_document"]}
    ordered = [d["pmc_id"] for d in baseline_rows]
    remaining = [i for i in ordered if i not in set(pilot)]
    require(len(remaining) == 72, "Non-pilot comparison must contain exactly 72 documents.")
    report = {
        "valid": True,
        "method": args.method,
        "files": full["files"],
        "inference": {key: summary.get(key) for key in (
            "model", "documents", "gpu_documents", "cpu_documents", "all_documents_refined",
            "elapsed_seconds", "entities_unchanged", "source_gold_annotations_used",
            "target_gold_annotations_used", "retrieval_training_annotations_used")},
        "summary_path": str(args.summary.resolve()),
        "selection_path": str(args.selection.resolve()),
        "selection_sha256": digest_file(args.selection),
        "pilot_ids": pilot,
        "validation": full["validation"],
        "caveat": (
            "All-80 scores are reused-corpus diagnostics. Non-pilot-72 scores exclude the "
            "eight pilot documents, but do not by themselves establish independent validation "
            "when the corpus has been reused for model/method selection. No A-set gold is used."
        ),
        "scopes": {
            "all_80": score_scope(ordered, gold, baseline, candidate, per_document),
            "non_pilot_72": score_scope(remaining, gold, baseline, candidate, per_document),
        },
    }
    require(digest_file(args.candidate) == candidate_hash,
            "Candidate changed while scores were being computed.")
    require(digest_file(args.baseline) == full["files"]["baseline"]["sha256"],
            "Fixed CPU baseline changed while scores were being computed.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("plain_gpu", "rag"), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl")
    parser.add_argument("--baseline", type=Path, default=ROOT / "work/gpu_route/cpu_oof_input.jsonl")
    parser.add_argument("--selection", type=Path, default=ROOT / "work/gpu_route/validation_selection.json")
    args = parser.parse_args()
    protected = {p.resolve() for p in (args.gold, args.baseline, args.candidate,
                                       args.summary, args.selection, Path(__file__))}
    require(args.output.resolve() not in protected, "Output must not overwrite an input or this script.")
    try:
        report = summarize(args)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = {"valid": False, "method": args.method, "errors": [f"{type(exc).__name__}: {exc}"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console = {"valid": report["valid"], "method": args.method}
    if report["valid"]:
        console["scopes"] = {
            name: {"baseline_score": value["baseline"]["score"],
                   "candidate_score": value["candidate"]["score"],
                   "score_delta": value["delta"]["score"],
                   "document_outcomes": value["document_outcomes"]}
            for name, value in report["scopes"].items()
        }
    else:
        console["errors"] = report["errors"]
    print(json.dumps(console, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
