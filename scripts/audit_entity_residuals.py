"""Attribute entity errors from frozen development predictions without fitting."""

from collections import Counter, defaultdict
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.evaluation import _positive_units, evaluate


def run(prediction_path=None, destination=None):
    source = ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl"
    prediction_path = (prediction_path or ROOT / "reports/association_context/best_development_oof.jsonl").resolve()
    documents = read_jsonl(source)
    predicted = read_jsonl(prediction_path)
    index = {d["pmc_id"]: d for d in predicted}
    counts = Counter()
    examples = defaultdict(list)
    phrases = defaultdict(Counter)
    for document in documents:
        p = index[document["pmc_id"]]
        gold = _positive_units(document, document["full_text"])
        pred = _positive_units(p, document["full_text"])
        mapped = {unit for unit in gold if unit[2] != "-1"}
        missing = mapped - pred
        remaining = pred - mapped
        for unit in sorted(gold - mapped):
            matches = sorted(v for v in remaining if v[:2] == unit[:2])
            if matches:
                remaining.remove(next((v for v in matches if v[2] == "-1"), matches[0]))
            else:
                missing.add(unit)
        negated = {(e["offset"], e["length"]) for e in p["entities"] if e.get("note") == "NO"}
        for kind, units, other in (("fn", missing, pred), ("fp", remaining, gold)):
            for start, length, concept in units:
                overlapping = [unit for unit in other if start < unit[0] + unit[1] and unit[0] < start + length]
                if kind == "fn" and (start, length) in negated:
                    category = "false_negated_exact"
                elif any(unit[:2] == (start, length) for unit in overlapping):
                    category = "exact_span_wrong_concept"
                elif any(unit[2] == concept for unit in overlapping):
                    category = "overlap_same_concept"
                elif overlapping:
                    category = "overlap_different_concept"
                else:
                    category = "no_overlap"
                key = kind + ":" + category
                counts[key] += 1
                paragraph = next(par for par in document["full_text"] if par["offset"] <= start < par["offset"] + len(par["text"]))
                relative = start - paragraph["offset"]
                text = paragraph["text"][relative:relative + length]
                phrases[key][text] += 1
                if len(examples[key]) < 20:
                    examples[key].append({"pmc_id": document["pmc_id"], "offset": start, "length": length,
                                          "text": text, "concept": concept, "overlap": sorted(overlapping),
                                          "context": paragraph["text"][max(0, relative-100):relative+length+100]})
    metrics = evaluate(documents, predicted)
    assert sum(value for key, value in counts.items() if key.startswith("fn:")) == metrics["mention"]["fn"]
    assert sum(value for key, value in counts.items() if key.startswith("fp:")) == metrics["mention"]["fp"]
    result = {"input_sha256": {str(p.relative_to(ROOT)): digest_file(p) for p in (source, prediction_path)},
              "code_sha256": digest_file(Path(__file__)), "metrics": metrics, "counts": dict(counts),
              "frequent_phrases": {key: value.most_common(30) for key, value in phrases.items()},
              "examples": dict(examples), "scope": "Diagnostic on reused development documents; not a new validation score."}
    destination = destination or ROOT / "reports/entity_residuals/audit.json"
    destination.parent.mkdir(exist_ok=True)
    write_json(destination, result)
    print(dict(counts))
    for key in sorted(phrases):
        print(key, phrases[key].most_common(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.predictions, args.output)
