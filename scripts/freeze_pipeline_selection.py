"""Bind the development choice to training parameters, resources and code."""

import argparse
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The pipeline choice is immutable once frozen")
    directory = args.development_dir.resolve()
    comparison_path, config_path = directory / "comparison.json", directory / "config.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if digest_file(comparison_path) != config["artifact_sha256"]["comparison.json"]:
        raise ValueError("Development comparison differs from its recorded artifact hash")
    manifest_path = Path(config["paths"]["manifest"])
    if digest_file(manifest_path) != config["input_sha256"]["manifest"]:
        raise ValueError("Development training manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parameters = manifest["parameters"]
    defaults = {"boundary_ranking_weight": 0.0, "boundary_ranking_margin": 1.0}
    names = ("epochs", "batch_size", "gradient_accumulation", "learning_rate", "head_learning_rate",
             "weight_decay", "warmup_ratio", "gradient_clip", "pos_weight_cap", "max_length", "stride",
             "max_span_width", "head_rank", "dropout", "seed", "fp16", "model_source", "model_revision",
             "boundary_ranking_weight", "boundary_ranking_margin")
    expected = {name: parameters.get(name, defaults.get(name)) for name in names}
    if any(value is None for value in expected.values()):
        raise ValueError("Missing a necessary selected training parameter")
    code = ("scripts/evaluate_linked_pipeline.py", "scripts/finalize_b_pipeline.py", "scripts/predict_span_ner.py",
            "patientphex/span_linking.py", "patientphex/patient_linking.py", "patientphex/association.py",
            "patientphex/entities.py", "patientphex/ontology.py", "patientphex/validation.py", "patientphex/data.py")
    result = {
        "schema_version": 1, "phase": "B", "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selected": comparison["selected"], "ner_training_parameters": expected,
        "model_files_sha256": manifest["sha256"]["model_files"],
        "resource_sha256": {"training_data": config["input_sha256"]["data"],
                            **{name: config["input_sha256"][name] for name in ("ontology", "umls_terms", "umls_manifest")}},
        "association_parameters": {"C": 0.5, "negative_weight": config["negative_weight"]},
        "inference_code_sha256": {name: digest_file(ROOT / name) for name in code},
        "selection_sources": {name: {"path": str(path), "sha256": digest_file(path)} for name, path in (
            ("comparison", comparison_path), ("configuration", config_path), ("training_manifest", manifest_path))},
        "decision": "Highest complete four-metric score on development fold 0 among the original run and two predeclared boundary controls; differences are not a significance claim",
        "confirmation": "Remaining folds use these fixed span/entity/association choices and final epoch 5. Training and development overlap across folds, so the overall result is a development estimate, not an independent test.",
        "target": "PatientPheX-V1-B/PatientPheX-V1-B.jsonl; never used to select parameters",
    }
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": digest_file(args.output), "selected": result["selected"]["name"]}))


if __name__ == "__main__":
    main()
