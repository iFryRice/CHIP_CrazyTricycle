"""Compare association refinements against a fixed CPU baseline without mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from patientphex.data import digest_file, read_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.validation import validate_submission


METRICS = ("mention", "document", "association_micro", "association_macro")


def comparison(gold, baseline, candidate):
    before = evaluate(gold, baseline)
    after = evaluate(gold, candidate)
    delta = {
        name: {key: after[name][key] - before[name][key]
               for key in ("precision", "recall", "f1")}
        for name in METRICS
    }
    delta["score"] = after["score"] - before["score"]
    return {"baseline": before, "candidate": after, "delta": delta}


def build_report(gold_path, baseline_path, candidate_path):
    gold_path, baseline_path, candidate_path = map(
        Path, (gold_path, baseline_path, candidate_path)
    )
    gold = read_jsonl(gold_path)
    baseline = read_jsonl(baseline_path)
    candidate = read_jsonl(candidate_path)
    indexed = [{d["pmc_id"]: d for d in rows}
               for rows in (gold, baseline, candidate)]
    gold_by_id, baseline_by_id, candidate_by_id = indexed
    gold_ids, baseline_ids, candidate_ids = map(set, indexed)
    coverage = {
        "gold_documents": len(gold),
        "baseline_documents": len(baseline),
        "candidate_documents": len(candidate),
        "candidate_missing_baseline_documents": sorted(baseline_ids - candidate_ids),
        "candidate_unexpected_documents": sorted(candidate_ids - baseline_ids),
        "baseline_documents_not_in_gold": sorted(baseline_ids - gold_ids),
        "candidate_documents_not_in_gold": sorted(candidate_ids - gold_ids),
        "gold_documents_outside_this_comparison": sorted(gold_ids - baseline_ids),
    }
    full = baseline_ids == gold_ids and bool(gold_ids)
    report = {
        "valid": False,
        "scope": {
            "kind": "full_training_set" if full else "pilot_subset",
            "documents": len(baseline),
            "is_complete_80_document_training_set": full and len(gold) == 80,
            "is_independent_validation": False,
            "caveat": (
                "A selected pilot or a reused training-set diagnostic is not an "
                "independent validation score. These local metrics are not the "
                "official A-set leaderboard score; no A-set gold is consulted."
            ),
        },
        "coverage": coverage,
        "files": {
            name: {"path": str(path.resolve()), "sha256": digest_file(path)}
            for name, path in (("gold", gold_path), ("baseline", baseline_path),
                               ("candidate", candidate_path))
        },
        "errors": [],
    }
    errors = report["errors"]
    if not baseline:
        errors.append("Baseline comparison set is empty.")
    for key in (
        "candidate_missing_baseline_documents", "candidate_unexpected_documents",
        "baseline_documents_not_in_gold", "candidate_documents_not_in_gold",
    ):
        if coverage[key]:
            errors.append(f"{key}: {coverage[key]}")
    for pmc_id in sorted(baseline_ids & candidate_ids):
        before, after = baseline_by_id[pmc_id], candidate_by_id[pmc_id]
        if before.get("pmid") != after.get("pmid"):
            errors.append(f"{pmc_id}: candidate changed PMID.")
        if before.get("entities") != after.get("entities"):
            errors.append(f"{pmc_id}: candidate changed entities or their ordering.")
    if errors:
        return report

    selected_gold = [gold_by_id[d["pmc_id"]] for d in baseline]
    if any(not d.get("patient") or not d.get("association") for d in selected_gold):
        errors.append("Gold must contain patients and labelled associations; unlabeled sets cannot be scored.")
        return report
    ontology = Ontology(gold_path.parent / "hp.obo")
    report["validation"] = {}
    for name, path in (("baseline", baseline_path), ("candidate", candidate_path)):
        try:
            report["validation"][name] = validate_submission(path, selected_gold, ontology)
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        return report

    ordered_candidate = [candidate_by_id[d["pmc_id"]] for d in baseline]
    report["overall"] = comparison(selected_gold, baseline, ordered_candidate)
    report["groups"] = {}
    report["per_document"] = []
    for name, predicate in (("single_patient", lambda n: n == 1),
                            ("multiple_patients", lambda n: n > 1)):
        ids = [d["pmc_id"] for d in selected_gold if predicate(len(d["patient"]))]
        report["groups"][name] = (
            comparison([gold_by_id[i] for i in ids], [baseline_by_id[i] for i in ids],
                       [candidate_by_id[i] for i in ids])
            if ids else {"documents": 0, "comparison": None}
        )
    for document in selected_gold:
        pmc_id = document["pmc_id"]
        scores = comparison([document], [baseline_by_id[pmc_id]], [candidate_by_id[pmc_id]])
        report["per_document"].append({
            "pmc_id": pmc_id,
            "patients": len(document["patient"]),
            "association_changed": baseline_by_id[pmc_id]["association"] != candidate_by_id[pmc_id]["association"],
            "baseline": {key: scores["baseline"][key] for key in ("association_micro", "association_macro")},
            "candidate": {key: scores["candidate"][key] for key in ("association_micro", "association_macro")},
            "delta": {key: scores["delta"][key] for key in ("association_micro", "association_macro", "score")},
        })
    report["changed_documents"] = sum(d["association_changed"] for d in report["per_document"])
    report["valid"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protected = {p.resolve() for p in (args.gold, args.baseline, args.candidate,
                                       args.gold.parent / "hp.obo", Path(__file__))}
    if args.output.resolve() in protected:
        parser.error("Output must not overwrite any input, ontology, or this script.")
    try:
        report = build_report(args.gold, args.baseline, args.candidate)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {key: report[key] for key in ("valid", "scope", "coverage", "errors") if key in report}
    if report["valid"]:
        summary["scores"] = {"baseline": report["overall"]["baseline"]["score"],
                             "candidate": report["overall"]["candidate"]["score"],
                             "delta": report["overall"]["delta"]["score"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
