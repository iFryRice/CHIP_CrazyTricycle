"""Reconstruct the final B candidate from persisted probabilities and audit protection."""

from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.llm_concept_review import eligible
from patientphex.ontology import Ontology
from patientphex.span_linking import SpanLinker
from patientphex.validation import validate_submission
from scripts.optimize_cpu_association import from_scores


def canonical(entity):
    return json.dumps(entity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def run():
    plan_path = ROOT / "experiments/joint_b/plan.json"
    plan = json.loads(plan_path.read_text())
    report, work = ROOT / plan["report_dir"], ROOT / plan["work_dir"]
    summary = json.loads((report / "summary.json").read_text())
    state = json.loads((report / "queue.json").read_text())
    if (state["status"] != "completed" or summary["status"] != "completed"
            or summary["plan_sha256"] != digest_file(plan_path) or state["sha256"] != summary["sha256"]):
        raise ValueError("The B controller and candidate must both be completed under the frozen plan.")
    for key in ["train_path", "target_path", "baseline_b", "ontology_path", "confirmation_plan", "confirmation_summary", "concept_plan"]:
        name = plan[key]
        if digest_file(ROOT / name) != plan["input_sha256"][name]:
            raise ValueError(f"A frozen source changed: {name}")
    source = read_jsonl(ROOT / plan["target_path"])
    old = read_jsonl(ROOT / plan["baseline_b"])
    new = read_jsonl(ROOT / plan["output"])
    ontology = Ontology(ROOT / plan["ontology_path"])
    validation = validate_submission(ROOT / plan["output"], source, ontology)
    if validation["sha256"] != summary["sha256"] or not validation["document_order_preserved"]:
        raise ValueError("The output content or document order differs from the verified candidate.")
    if [d["pmc_id"] for d in old] != [d["pmc_id"] for d in new]:
        raise ValueError("B document identities changed during inference.")
    fixed = SpanLinker(ontology)
    removed, protected = 0, 0
    removed_records = []
    for baseline, candidate in zip(old, new, strict=True):
        before, after = Counter(map(canonical, baseline["entities"])), Counter(map(canonical, candidate["entities"]))
        if after - before:
            raise ValueError("Joint B changed or added an original entity record.")
        preserved = Counter(canonical(e) for e in baseline["entities"] if not eligible(e, ontology, fixed))
        if preserved - after:
            raise ValueError("Joint B removed a protected original entity.")
        protected += sum(preserved.values())
        removed += sum((before-after).values())
        removed_records.extend({"pmc_id": baseline["pmc_id"], "entity": json.loads(key)} for key, count in (before-after).items() for _ in range(count))
    if removed != summary["removed_entities"]:
        raise ValueError("Entity edit count differs from the inference report.")
    identities = json.loads((work / "example_identity.json").read_text())
    keys = [(r["pmc_id"], r["patient_id"], r["concept"]) for r in identities]
    members = list(plan["text_models"])
    if len(set(keys)) != len(keys) or set(summary["text_fold_probability_sha256"]) != set(members) or len(members) != 15:
        raise ValueError("B relation keys or ensemble coverage are incomplete.")
    totals = [0.0] * len(keys)
    for member in members:
        path = work / f"{member}_probabilities.json"
        if digest_file(path) != summary["text_fold_probability_sha256"][member]:
            raise ValueError("A saved member probability file changed.")
        values = json.loads(path.read_text())
        if len(values) != len(keys) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("A saved probability sequence is invalid or incomplete.")
        totals = [a+b/len(members) for a, b in zip(totals, values, strict=True)]
    probabilities = dict(zip(keys, totals, strict=True))
    context = json.loads((work / "context_scores.json").read_text())
    entities = json.loads((work / "filtered_entities.json").read_text())
    expected_keys = {(identifier, r["patient_id"], r["concept"]) for identifier, rows in context.items() for r in rows}
    if expected_keys != set(keys):
        raise ValueError("Structural and text model candidate coverage differs.")
    replay = []
    for document in source:
        identifier = str(document["pmc_id"])
        scores = [{**row, "score": 0.75*row["score"]+0.25*probabilities[identifier, row["patient_id"], row["concept"]]}
                  for row in context[identifier]]
        replay.append(from_scores(document, entities[identifier], scores, 0.5))
    if replay != new:
        raise ValueError("The final candidate does not exactly replay from all 15 member outputs.")
    destination = ROOT / "reports/joint_b_verification"
    write_json(destination / "removed_entities.json", removed_records)
    result = {"status": "passed", "candidate_sha256": summary["sha256"], "validation": validation,
              "protected_entities_preserved": protected, "removed_entity_records": removed,
              "all_retained_entities_exact_original_records": True, "text_members_verified": len(members),
              "source_pairs": len(keys), "final_probabilities_and_prediction_exact_replay": True,
              "plan_sha256": digest_file(plan_path), "code_sha256": digest_file(Path(__file__)),
              "scope": "Engineering verification of the complete blind B artifact, not an accuracy estimate or platform submission."}
    write_json(destination / "summary.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    run()
