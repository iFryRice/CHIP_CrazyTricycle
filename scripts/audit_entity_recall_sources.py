"""Locate missing gold spans in saved candidates; diagnostic only, never predict."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.entities import phrase_key
from patientphex.evaluation import _positive_units, evaluate
from patientphex.ontology import Ontology


def missing_units(gold, proposed):
    mapped = {unit for unit in gold if unit[2] != "-1"}
    missing, remaining = mapped - proposed, proposed - mapped
    for unit in sorted(gold - mapped):
        matches = sorted(value for value in remaining if value[:2] == unit[:2])
        if matches:
            remaining.remove(next((value for value in matches if value[2] == "-1"), matches[0]))
        else:
            missing.add(unit)
    return missing


def run():
    paths = {"source": "PatientPheX-V1-A/PatientPheX-train.jsonl", "split": "reports/cpu_baseline/split.json",
             "ontology": "PatientPheX-V1-A/hp.obo", "predictions": "reports/joint_confirmation/development_oof.jsonl",
             "baseline": "reports/association_context/best_development_oof.jsonl",
             "semantic_summary": "reports/semantic_linking/summary.json", "semantic_plan": "experiments/semantic_linking/plan.json"}
    docs = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / paths["source"])}
    before = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / paths["baseline"])}
    after = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / paths["predictions"])}
    folds = json.loads((ROOT / paths["split"]).read_text())["folds"]
    summary = json.loads((ROOT / paths["semantic_summary"]).read_text())
    plan = json.loads((ROOT / paths["semantic_plan"]).read_text())
    selected = summary["selected"]
    ontology = Ontology(ROOT / paths["ontology"])
    hpo_keys = {phrase_key(text) for identifier in ontology.allowed_ids for text in
                [ontology.terms[identifier]["name"], *ontology.terms[identifier]["exact_synonyms"]]}
    counts, support, cross = Counter(), Counter(), defaultdict(Counter)
    examples = defaultdict(list)
    for fold, identifiers in enumerate(folds):
        candidate_path = f"reports/semantic_linking/fold{fold}/candidates.json"
        paths[f"fold{fold}_candidates"] = candidate_path
        candidates = json.loads((ROOT / candidate_path).read_text())
        if set(candidates) != set(identifiers):
            raise ValueError("Saved candidate coverage differs from its declared validation fold.")
        training_keys = {phrase_key(e["text"]) for identifier, d in docs.items() if identifier not in identifiers
                         for e in d["entities"] if e.get("note") != "NO"}
        for identifier in identifiers:
            document, baseline, prediction = docs[identifier], before[identifier], after[identifier]
            gold = _positive_units(document, document["full_text"])
            proposed = _positive_units(prediction, document["full_text"])
            previous = _positive_units(baseline, document["full_text"])
            old_missing = missing_units(gold, previous)
            for unit in sorted(missing_units(gold, proposed)):
                offset, length, concept = unit
                paragraph = next(p for p in document["full_text"] if p["offset"] <= offset and offset+length <= p["offset"]+len(p["text"]))
                text = paragraph["text"][offset-paragraph["offset"]:offset-paragraph["offset"]+length]
                key = phrase_key(text)
                lexical = "seen_in_fold_training" if key in training_keys else "official_hpo_only" if key in hpo_keys else "unseen_in_both"
                exact = [item for item in candidates[identifier] if (item["entity"]["offset"], item["entity"]["length"]) == unit[:2]
                         and item["entity"].get("note") != "NO"
                         and (concept == "-1" or concept in item["entity"]["identifier"].split(";"))]
                passing = [item for item in exact if item["span_score"] >= selected["span_threshold"]
                           and (item["source"] != "semantic" or item["similarity"] >= selected["semantic_threshold"]
                                and item["margin"] >= plan["margin_threshold"])]
                overlap = [e for e in baseline["entities"] if offset < e["offset"]+e["length"] and e["offset"] < offset+length]
                if unit not in old_missing:
                    category = "lost_after_concept_review"
                elif passing and overlap:
                    category = "passing_exact_candidate_blocked_by_overlap"
                elif passing:
                    category = "passing_exact_candidate_without_overlap_requires_investigation"
                elif exact:
                    category = "exact_candidate_below_frozen_confidence_gates"
                elif any((item["entity"]["offset"], item["entity"]["length"]) == unit[:2] for item in candidates[identifier]):
                    category = "saved_exact_span_but_wrong_mapping_or_negation"
                else:
                    category = "no_exact_span_in_saved_high_confidence_candidates"
                counts[category] += 1
                support[lexical] += 1
                cross[category][lexical] += 1
                if len(examples[category]) < 10:
                    examples[category].append({"pmc_id": identifier, "fold": fold, "text": text, "offset": offset,
                                               "length": length, "concept": concept, "lexical_support": lexical,
                                               "exact_candidates": exact, "overlapping_baseline_entities": overlap})
    metrics = evaluate(docs.values(), after.values())
    assert sum(counts.values()) == metrics["mention"]["fn"]
    result = {"scope": "Label-using attribution only. Saved neural proposals have score >= 0.9; absence is not proof that the neural model never generated a span. Candidate availability is not measured accuracy or an attainable score. No predictions are exported.",
              "input_sha256": {name: digest_file(ROOT / name) for name in paths.values()}, "code_sha256": digest_file(Path(__file__)),
              "saved_candidate_minimum_span_score": 0.9, "selected_original_configuration": selected,
              "missing_mentions": metrics["mention"]["fn"], "candidate_categories": dict(counts),
              "fold_training_and_hpo_surface_support": dict(support), "cross_tabulation": dict(cross), "examples": dict(examples)}
    write_json(ROOT / "reports/entity_residuals/joint_recall_sources.json", result)
    print(json.dumps({key: result[key] for key in ["missing_mentions", "candidate_categories", "fold_training_and_hpo_surface_support", "cross_tabulation"]}))


if __name__ == "__main__":
    run()
