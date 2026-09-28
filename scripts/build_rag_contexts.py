"""Build auditable, fold-isolated HPO and training-example RAG contexts."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "PatientPheX-V1-A"
OUTPUT = ROOT / "work/gpu_route/rag_contexts.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_digest(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def load_jsonl(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    mapped = {str(record["pmc_id"]): record for record in records}
    assert len(mapped) == len(records), f"Duplicate pmc_id in {path}"
    return mapped


def raw_text(doc):
    return "\n".join(part["text"] for part in doc["full_text"])


def hpo_ids(value):
    return set(re.findall(r"HP:\d{7}", str(value)))


def load_hpo(path):
    terms = {}
    version = None
    current = None
    quoted = re.compile(r'^\w+: "((?:\\.|[^"\\])*)"')
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("data-version: "):
            version = line.split(": ", 1)[1]
        if line.startswith("["):
            current = {} if line == "[Term]" else None
        elif current is not None and line.startswith("id: "):
            current["id"] = line[4:]
            terms[current["id"]] = current
        elif current is not None and line.startswith("name: "):
            current["name"] = line[6:]
        elif current is not None and line.startswith("is_a: "):
            current.setdefault("parents", []).append(line[6:].split()[0])
        elif current is not None and line.startswith("is_obsolete: true"):
            current["obsolete"] = True
        elif current is not None and line.startswith(("def: ", "synonym: ")):
            match = quoted.match(line)
            if match:
                value = match.group(1).replace('\\"', '"').replace('\\n', ' ')
                if line.startswith("def: "):
                    current["definition"] = value[:300]
                else:
                    current.setdefault("synonyms", []).append(value)
    branch = {"HP:0000118"}
    while True:
        extra = {key for key, term in terms.items() if not term.get("obsolete") and set(term.get("parents", [])) & branch} - branch
        if not extra:
            break
        branch.update(extra)
    return terms, branch, version


def evidence_window(doc, entity, radius=160):
    offset = int(entity["offset"])
    for part in doc["full_text"]:
        start = int(part["offset"])
        text = part["text"]
        if start <= offset < start + len(text):
            local = offset - start
            left = max(0, local - radius)
            right = min(len(text), local + int(entity.get("length", len(entity["text"]))) + radius)
            return {"offset": start + left, "text": text[left:right]}
    return None


def make_example(doc, candidate_ids, score):
    patients = {p["patient_id"]: p for p in doc["patient"]}
    relations = []
    for association in doc["association"]:
        patient_id = association["patient_id"]
        patient = patients.get(patient_id)
        if patient is None:
            continue
        mentions = patient.get("mention", [])
        gold_ids = set().union(*(hpo_ids(item) for item in association.get("phenotype", [])))
        for hpo_id in sorted(gold_ids & candidate_ids):
            evidence = [e for e in doc["entities"] if hpo_id in hpo_ids(e.get("identifier", "")) and e.get("note") != "NO"]
            if not evidence:
                continue
            evidence.sort(key=lambda e: min((abs(int(e["offset"]) - int(m["offset"])) for m in mentions), default=0))
            entity = evidence[0]
            window = evidence_window(doc, entity)
            if window is None:
                continue
            near_mentions = sorted(mentions, key=lambda m: abs(int(m["offset"]) - int(entity["offset"])))[:2]
            relations.append({"patient_id": patient_id, "patient_mentions": near_mentions, "phenotype": hpo_id, "gold_association": True, "phenotype_mention": {key: entity.get(key) for key in ("text", "offset", "length", "note")}, "evidence_window": window})
            if len(relations) == 2:
                break
        if len(relations) == 2:
            break
    if not relations:
        return None
    example = {"pmc_id": str(doc["pmc_id"]), "text_cosine_similarity": round(float(score), 6), "relations": relations}
    while len(json.dumps(example, ensure_ascii=False)) > 2500 and len(example["relations"]) > 1:
        example["relations"].pop()
    if len(json.dumps(example, ensure_ascii=False)) > 2500:
        return None
    return example


def main():
    paths = {"train": DATA / "PatientPheX-train.jsonl", "a": DATA / "PatientPheX-A.jsonl", "hpo": DATA / "hp.obo", "split": ROOT / "reports/cpu_baseline/split.json", "cpu_oof": ROOT / "work/gpu_route/cpu_oof_input.jsonl", "cpu_a": ROOT / "work/gpu_route/cpu_a_input.jsonl"}
    train = load_jsonl(paths["train"])
    a_docs = load_jsonl(paths["a"])
    oof = load_jsonl(paths["cpu_oof"])
    a_predictions = load_jsonl(paths["cpu_a"])
    folds = json.loads(paths["split"].read_text(encoding="utf-8-sig"))["folds"]
    flat_ids = [str(pmc_id) for fold in folds for pmc_id in fold]
    assert len(folds) == 5 and len(flat_ids) == len(set(flat_ids)) == len(train) == 80
    assert set(flat_ids) == set(train) == set(oof)
    assert len(a_docs) == 20 and set(a_docs) == set(a_predictions) and not set(a_docs) & set(train)
    terms, branch, version = load_hpo(paths["hpo"])
    assert version == "hp/releases/2026-06-23", version
    contexts = {}
    targets = [(fold, {key: train[key] for key in fold}, oof) for fold in folds]
    targets.append(([], a_docs, a_predictions))
    for excluded, documents, predictions in targets:
        excluded = {str(item) for item in excluded}
        eligible = sorted(set(train) - excluded)
        assert not set(eligible) & excluded
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True, max_features=40000)
        matrix = vectorizer.fit_transform([raw_text(train[key]) for key in eligible])
        for pmc_id, doc in documents.items():
            candidates = set().union(*(hpo_ids(entity.get("identifier", "")) for entity in predictions[pmc_id].get("entities", []) if entity.get("note") != "NO"))
            knowledge = {}
            for hpo_id in sorted(candidates):
                term = terms.get(hpo_id, {})
                knowledge[hpo_id] = {"name": term.get("name", ""), "definition": term.get("definition", "")[:300], "synonyms": [value[:120] for value in term.get("synonyms", [])[:3]], "in_phenotypic_abnormality_branch": hpo_id in branch and not term.get("obsolete", False)}
            scores = (matrix @ vectorizer.transform([raw_text(doc)]).T).toarray().ravel()
            selected = sorted(range(len(eligible)), key=lambda index: (-float(scores[index]), eligible[index]))[:2]
            retrieved = [eligible[index] for index in selected]
            assert pmc_id not in retrieved and not set(retrieved) & excluded
            examples = [example for index in selected if (example := make_example(train[eligible[index]], candidates, scores[index])) is not None]
            contexts[pmc_id] = {"hpo_candidates": knowledge, "examples": examples, "excluded_fold_ids": sorted(excluded), "retrieved_ids": [example["pmc_id"] for example in examples], "ranked_retrieval_ids": retrieved, "eligible_training_ids_hash": ids_digest(eligible), "eligible_training_count": len(eligible)}
    assert len(contexts) == 100
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "hpo_version": version, "hpo_root": "HP:0000118", "fold_exclusion_rule": "For each training target, exclude all 16 documents in its heldout fold from vectorizer fitting and retrieval; A targets use all 80 training documents.", "query_rule": "TF-IDF uses only original full_text; target gold entities and associations are never used. Candidate HPO IDs come only from supplied CPU prediction snapshots.", "example_rule": "Retrieve the top two documents by text cosine similarity only. Include an example only if its training gold association overlaps target predicted HPO IDs and has non-negated entity evidence. No replacement by lower-ranked documents.", "evidence_rule": "Gold patient-phenotype labels are authoritative; the nearby phenotype window illustrates original wording and is not separately annotated pair-level evidence.", "example_char_limit": 2500, "definition_char_limit": 300, "fold_sizes": [len(fold) for fold in folds], "source_sha256": {name: digest(path) for name, path in paths.items()}, "script_sha256": digest(Path(__file__)), "document_count": len(contexts), "example_count": sum(len(context["examples"]) for context in contexts.values())}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"metadata": metadata, "contexts": contexts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "documents": len(contexts), "examples": metadata["example_count"], "split_sha256": metadata["source_sha256"]["split"], "fold_isolation": "passed"}))


if __name__ == "__main__":
    main()
