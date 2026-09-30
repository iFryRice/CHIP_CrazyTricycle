"""Measure an explicitly label-using diagnostic ceiling; never export predictions."""

import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.association import _concepts, _gold_concepts
from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.evaluation import evaluate
from scripts.optimize_cpu_association import error_counts


def run():
    train = ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl"
    baseline_path = ROOT / "reports/association_context/best_development_oof.jsonl"
    documents = read_jsonl(train)
    baseline = read_jsonl(baseline_path)
    gold = {str(document["pmc_id"]): {a["patient_id"]: _gold_concepts(a["phenotype"]) for a in document["association"]} for document in documents}
    veto_oracle, reachable_oracle = copy.deepcopy(baseline), copy.deepcopy(baseline)
    for prediction in veto_oracle:
        targets = gold[str(prediction["pmc_id"])]
        for relation in prediction["association"]:
            relation["phenotype"] = [concept for concept in relation["phenotype"]
                                     if not concept.startswith("HP:") or concept in targets[relation["patient_id"]]]
    for prediction in reachable_oracle:
        targets = gold[str(prediction["pmc_id"])]
        candidates = {concept for entity in prediction["entities"] for concept in _concepts(entity)}
        for relation in prediction["association"]:
            relation["phenotype"] = sorted(candidates & targets[relation["patient_id"]])
    result = {"scope": "LABEL-USING ORACLE DIAGNOSTIC, not a trained model, cross-validation improvement, independent test, or B score. Never emit its predictions.",
              "input_sha256": {str(train.relative_to(ROOT)): digest_file(train), str(baseline_path.relative_to(ROOT)): digest_file(baseline_path)},
              "baseline": {"metrics": evaluate(documents, baseline), "errors": error_counts(documents, baseline)},
              "remove_only_false_mapped_existing_relations_oracle": {"metrics": evaluate(documents, veto_oracle), "errors": error_counts(documents, veto_oracle)},
              "perfect_relation_labels_among_existing_entity_concepts_oracle": {"metrics": evaluate(documents, reachable_oracle), "errors": error_counts(documents, reachable_oracle)},
              "use": "Distinguish headroom in false-link removal from missing-entity limitations; no label-derived thresholds or decisions enter inference."}
    write_json(ROOT / "reports/entity_residuals/association_ceiling_diagnostic.json", result)
    print({key: value["metrics"]["score"] for key, value in result.items() if isinstance(value, dict) and "metrics" in value})


if __name__ == "__main__":
    run()
