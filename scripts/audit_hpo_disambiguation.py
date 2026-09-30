"""Quantify cached concept alternatives using labels only for diagnostics."""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.evaluation import _positive_units, is_evaluation_passage
from patientphex.ontology import Ontology
from patientphex.span_linking import SpanLinker, _strict_key


def run():
    train = ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl"
    predicted = ROOT / "reports/association_context/best_development_oof.jsonl"
    retrieved_path = ROOT / "work/span_ner/semantic_linking/development_retrieval.json"
    documents = read_jsonl(train)
    baseline = {str(d["pmc_id"]): d for d in read_jsonl(predicted)}
    retrieved = json.loads(retrieved_path.read_text())
    ontology = Ontology(ROOT / "PatientPheX-V1-A/hp.obo")
    fixed = SpanLinker(ontology)
    totals, folds, examples = Counter(), {}, []
    partitions = json.loads((ROOT / "reports/cpu_baseline/split.json").read_text())["folds"]
    fold_lookup = {identifier: f for f, identifiers in enumerate(partitions) for identifier in identifiers}
    for document in documents:
        identifier = str(document["pmc_id"])
        fold = fold_lookup[identifier]
        local = folds.setdefault(str(fold), Counter())
        passages = [p for p in document["full_text"] if is_evaluation_passage(p)]
        gold = _positive_units(document, passages)
        for entity in baseline[identifier]["entities"]:
            current = entity["identifier"]
            if entity.get("note") == "NO" or ";" in current or current not in ontology.allowed_ids:
                continue
            counts = Counter(positive_single_hpo_entities=1)
            retrieval = retrieved.get(entity["text"])
            if retrieval is None:
                counts["missing_cached_retrieval"] += 1
                totals.update(counts); local.update(counts)
                continue
            counts["cached_retrieval_available"] += 1
            candidates = {row["identifier"] for row in retrieval["alternatives"]} | {current}
            protected = fixed.hpo_exact.get(_strict_key(entity["text"])) == {current}
            if protected:
                counts["unique_exact_hpo_protected"] += 1
            else:
                counts["reviewable_without_unique_exact_hpo"] += 1
            prefix = "protected" if protected else "reviewable"
            exact = {concept for start, length, concept in gold if (start, length) == (entity["offset"], entity["length"])}
            overlapping = {concept for start, length, concept in gold
                           if start < entity["offset"]+entity["length"] and entity["offset"] < start+length}
            if current in exact or "-1" in exact:
                counts[prefix+":exact_current_correct"] += 1
            elif exact & candidates:
                counts[prefix+":exact_wrong_gold_in_candidates"] += 1
                if not protected:
                    paragraph = next(p for p in document["full_text"] if p["offset"] <= entity["offset"] < p["offset"]+len(p["text"]))
                    offset = entity["offset"]-paragraph["offset"]
                    examples.append({"fold": fold, "pmc_id": identifier, "entity": entity,
                        "gold_exact_ids": sorted(exact), "alternatives": [{**row, "name": ontology.terms[row["identifier"]]["name"]} for row in retrieval["alternatives"]],
                        "current_name": ontology.terms[current]["name"],
                        "context": paragraph["text"][max(0,offset-150):offset+entity["length"]+150]})
            elif exact:
                counts[prefix+":exact_wrong_gold_not_in_candidates"] += 1
            elif overlapping & candidates - {current}:
                counts[prefix+":overlap_gold_in_alternatives_only"] += 1
            elif overlapping:
                counts[prefix+":overlap_other"] += 1
            else:
                counts[prefix+":no_gold_overlap"] += 1
            totals.update(counts); local.update(counts)
    result = {"scope": "LABEL-USING CANDIDATE-COVERAGE DIAGNOSTIC ONLY. No new inference score or prediction; coverage is limited to the preexisting SapBERT cache. Gold labels cannot select individual deployment cases.",
              "input_sha256": {p.relative_to(ROOT).as_posix(): digest_file(p) for p in [train, predicted, retrieved_path]},
              "candidate_rule": "Current mapped single HPO concept plus cached top three distinct SapBERT concepts. Protect exact unique official HPO names/synonyms matching the current concept; all decisions are independent of labels.",
              "totals": dict(totals), "folds": {k: dict(v) for k,v in folds.items()}, "reviewable_exact_correction_examples": examples}
    write_json(ROOT / "reports/entity_residuals/hpo_disambiguation_audit.json", result)
    print(json.dumps({"totals": totals, "folds": folds}))


if __name__ == "__main__":
    run()
