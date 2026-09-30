"""Explain the completed fixed LLM pilot without fitting or changing predictions."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.evaluation import _association_set, _mention_counts, _positive_units, evaluate


def matched_units(gold, predicted):
    """Identify the same matched prediction units as the existing scorer."""
    matched = {unit for unit in gold if unit[2] != "-1"} & predicted
    remaining = predicted - matched
    for offset, length, _ in sorted(unit for unit in gold if unit[2] == "-1"):
        candidates = sorted(unit for unit in remaining if unit[:2] == (offset, length))
        if candidates:
            chosen = next((unit for unit in candidates if unit[2] == "-1"), candidates[0])
            matched.add(chosen)
            remaining.remove(chosen)
    if len(matched) != _mention_counts(gold, predicted)[0]:
        raise ValueError("Diagnostic matching differs from the fixed scorer.")
    return matched


def entity_key(entity):
    return entity["offset"], entity["length"], entity["text"], entity["identifier"], entity.get("note")


def run():
    plan_path = ROOT / "experiments/llm_extraction/plan.json"
    plan = json.loads(plan_path.read_text())
    report = ROOT / plan["report_dir"]
    summary = json.loads((report / "summary.json").read_text())
    if summary["plan_sha256"] != digest_file(plan_path):
        raise ValueError("The completed pilot belongs to another frozen plan.")
    eligible = [row for row in summary["ranked"] if row["eligible"]]
    selected_row = eligible[0] if eligible else summary["ranked"][0]
    source = ROOT / plan["train_path"]
    documents = read_jsonl(source)
    lookup = {str(document["pmc_id"]): document for document in documents}
    predictions_path = report / "best_pilot_predictions.jsonl"
    predictions = read_jsonl(predictions_path)
    base_path = ROOT / plan["baseline_oof"]
    baseline = {str(document["pmc_id"]): document for document in read_jsonl(base_path)}
    splits = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    expected_ids = {identifier for fold in plan["pilot_folds"] for identifier in splits[fold]}
    if {str(document["pmc_id"]) for document in predictions} != expected_ids:
        raise ValueError("Pilot predictions do not cover the declared folds exactly.")
    pilot = [lookup[str(document["pmc_id"])] for document in predictions]
    original = [baseline[str(document["pmc_id"])] for document in predictions]
    base_metrics = evaluate(pilot, original)
    metrics = evaluate(pilot, predictions)
    if metrics != selected_row["metrics"] or base_metrics != summary["baseline"]:
        raise ValueError("Completed evaluation metrics do not reproduce.")
    candidates, inputs = {}, {}
    for fold in plan["pilot_folds"]:
        path = report / f"fold{fold}_candidates.json"
        candidates.update(json.loads(path.read_text()))
        inputs[str(path.relative_to(ROOT))] = digest_file(path)
    counts, sources, relations = Counter(), defaultdict(Counter), Counter()
    examples, document_deltas = defaultdict(list), []
    for predicted in predictions:
        identifier = str(predicted["pmc_id"])
        document, base = lookup[identifier], baseline[identifier]
        old_entities = {entity_key(entity) for entity in base["entities"]}
        new_entities = [entity for entity in predicted["entities"] if entity_key(entity) not in old_entities]
        if not old_entities <= {entity_key(entity) for entity in predicted["entities"]}:
            raise ValueError("The addition-only pilot removed or changed baseline entities.")
        counts["added_entity_records"] += len(new_entities)
        counts["added_negated_records"] += sum(entity.get("note") == "NO" for entity in new_entities)
        gold = _positive_units(document, document["full_text"])
        before = _positive_units(base, document["full_text"])
        after = _positive_units(predicted, document["full_text"])
        if not before <= after:
            raise ValueError("The addition-only pilot removed scored entity units.")
        matched = matched_units(gold, after)
        negative_spans = {(entity["offset"], entity["length"]) for entity in document["entities"] if entity.get("note") == "NO"}
        for unit in sorted(after - before):
            offset, length, concept = unit
            matches = [item for item in candidates[identifier] if item["entity"]["offset"] == offset
                       and item["entity"]["length"] == length and concept in item["entity"]["identifier"].split(";")]
            if not matches:
                raise ValueError("An added entity lacks a recorded LLM mapping candidate.")
            proposal = matches[0]
            if unit in matched:
                category = "correct"
            elif (offset, length) in negative_spans:
                category = "gold_negated"
            elif any(value[:2] == unit[:2] for value in gold):
                category = "exact_span_wrong_concept"
            elif any(offset < value[0] + value[1] and value[0] < offset + length for value in gold):
                category = "overlap_wrong_boundary"
            else:
                category = "no_gold_overlap"
            counts[category] += 1
            sources[proposal["source"]]["tp" if category == "correct" else "fp"] += 1
            if len(examples[category]) < 30:
                paragraph = next(p for p in document["full_text"] if p["offset"] <= offset and offset+length <= p["offset"]+len(p["text"]))
                relative = offset-paragraph["offset"]
                examples[category].append({"pmc_id": identifier, "entity": proposal["entity"], "source": proposal["source"],
                                           "similarity": proposal["similarity"], "margin": proposal["margin"],
                                           "context": paragraph["text"][max(0, relative-100):relative+length+100]})
        actual = {row["patient_id"]: _association_set(row["phenotype"]) for row in document["association"]}
        previous = {row["patient_id"]: _association_set(row["phenotype"]) for row in base["association"]}
        revised = {row["patient_id"]: _association_set(row["phenotype"]) for row in predicted["association"]}
        for patient_id, gold_concepts in actual.items():
            added, removed = revised[patient_id]-previous[patient_id], previous[patient_id]-revised[patient_id]
            relations["added_tp"] += len(added & gold_concepts)
            relations["added_fp"] += len(added - gold_concepts)
            relations["removed_tp"] += len(removed & gold_concepts)
            relations["removed_fp"] += len(removed - gold_concepts)
        document_deltas.append({"pmc_id": identifier, "added_entities": len(new_entities),
                                "score_delta": evaluate([document], [predicted])["score"] - evaluate([document], [base])["score"]})
    added_units = sum(sources[name]["tp"] + sources[name]["fp"] for name in sources)
    if counts["correct"] != metrics["mention"]["tp"] - base_metrics["mention"]["tp"]:
        raise ValueError("Entity true-positive changes fail reconciliation.")
    if added_units - counts["correct"] != metrics["mention"]["fp"] - base_metrics["mention"]["fp"]:
        raise ValueError("Entity false-positive changes fail reconciliation.")
    if relations["added_tp"] - relations["removed_tp"] != metrics["association_micro"]["tp"] - base_metrics["association_micro"]["tp"]:
        raise ValueError("Association true-positive changes fail reconciliation.")
    if relations["added_fp"] - relations["removed_fp"] != metrics["association_micro"]["fp"] - base_metrics["association_micro"]["fp"]:
        raise ValueError("Association false-positive changes fail reconciliation.")
    inputs.update({str(path.relative_to(ROOT)): digest_file(path) for path in [source, base_path, predictions_path, plan_path, report / "summary.json"]})
    result = {"configuration": selected_row["configuration"], "promoted_to_confirmation": summary["selected"] is not None,
              "input_sha256": inputs, "implementation_sha256": digest_file(Path(__file__)), "entity_counts": dict(counts),
              "added_scored_units": added_units, "added_unit_precision": counts["correct"] / added_units if added_units else None,
              "mapping_sources": {key: dict(value) for key, value in sources.items()}, "association_changes": dict(relations),
              "document_deltas": sorted(document_deltas, key=lambda row: row["score_delta"]), "examples": dict(examples),
              "scope": "Post-hoc error diagnosis on the completed, repeatedly used development pilot; not an official score or independent validation."}
    destination = report / "error_audit.json"
    if destination.exists() and json.loads(destination.read_text()) != result:
        raise FileExistsError("An earlier diagnostic differs; refusing silent replacement.")
    write_json(destination, result)
    print(json.dumps({key: result[key] for key in ["configuration", "entity_counts", "added_unit_precision", "mapping_sources", "association_changes"]}), flush=True)


if __name__ == "__main__":
    run()
