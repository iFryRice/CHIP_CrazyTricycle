"""Prepare, infer and evaluate a frozen source-grounded relation-review pilot."""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.llm_association import build_task, parse_decision, prepare_prompt, should_veto


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def load_plan(path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f"Frozen input/code changed: {name}")
    return plan


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def prepare(plan_path, plan):
    from patientphex.ontology import Ontology
    from patientphex.patient_linking import _check_provenance
    from scripts.optimize_cpu_association import blind, from_scores

    documents = read_jsonl(ROOT / plan["train_path"])
    lookup = {str(document["pmc_id"]): document for document in documents}
    splits = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    if sorted(identifier for fold in splits for identifier in fold) != sorted(lookup):
        raise ValueError("The fixed folds do not partition the training corpus.")
    baseline = {str(record["pmc_id"]): record for record in read_jsonl(ROOT / plan["baseline_oof"])}
    if set(baseline) != set(lookup):
        raise ValueError("Baseline coverage differs from the source corpus.")
    ontology = Ontology(ROOT / plan["ontology_path"])
    root = ROOT / plan["work_dir"]
    if root.exists():
        raise FileExistsError("Do not overwrite an earlier association review.")
    packs = []
    for fold in plan["pilot_folds"]:
        validation_ids = set(splits[fold])
        with (ROOT / plan["association_models"][str(fold)]).open("rb") as handle:
            model = pickle.load(handle)
        _check_provenance(model.provenance_, set(lookup) - validation_ids, validation_ids)
        tasks = []
        for identifier in splits[fold]:
            document = blind(lookup[identifier])
            entities = baseline[identifier]["entities"]
            scores = model.predict_scores(document, entities)
            if from_scores(document, entities, scores, 0.5) != baseline[identifier]:
                raise ValueError("Frozen patient-linker baseline failed exact replay.")
            for score in scores:
                if score["score"] >= 0.5 and score["concept"].startswith("HP:"):
                    tasks.append(build_task(document, entities, score["patient_id"], score["concept"], score["score"], ontology))
        if len({task["task_id"] for task in tasks}) != len(tasks):
            raise ValueError("Association review tasks are duplicated.")
        packs.append({"fold": fold, "tasks": tasks, "validation_document_ids": splits[fold],
                      "upstream_training_document_ids": sorted(set(lookup) - validation_ids),
                      "llm_training_document_ids": [], "demonstration_document_ids": [],
                      "plan_sha256": digest_file(plan_path), "baseline_exact_replay": True,
                      "answers_removed_before_prompting": True})
    for pack in packs:
        directory = root / f"fold{pack['fold']}"
        write_json(directory / "tasks.json", pack)
        manifest = {key: value for key, value in pack.items() if key != "tasks"}
        manifest.update(task_count=len(pack["tasks"]), tasks_sha256=digest_file(directory / "tasks.json"),
                        all_occurrences_selected=sum(task["all_occurrences_selected"] for task in pack["tasks"]))
        write_json(directory / "prepared.json", manifest)
        print(json.dumps(manifest), flush=True)


def load_tasks(plan_path, plan, fold):
    if fold not in plan["pilot_folds"]:
        raise ValueError("This plan authorizes only the declared folds.")
    directory = ROOT / plan["work_dir"] / f"fold{fold}"
    manifest = json.loads((directory / "prepared.json").read_text())
    if manifest["plan_sha256"] != digest_file(plan_path) or manifest["tasks_sha256"] != digest_file(directory / "tasks.json"):
        raise ValueError("Prepared tasks belong to another plan or were altered.")
    pack = json.loads((directory / "tasks.json").read_text())
    if pack["llm_training_document_ids"] or pack["demonstration_document_ids"] or not pack["answers_removed_before_prompting"]:
        raise ValueError("The declared zero-shot input boundary changed.")
    if set(pack["validation_document_ids"]) & set(pack["upstream_training_document_ids"]):
        raise ValueError("Upstream training and validation overlap.")
    return directory, manifest, pack["tasks"]


def infer(plan_path, plan, fold, preflight=False):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    directory, manifest, tasks = load_tasks(plan_path, plan, fold)
    spec = json.loads((ROOT / plan["environment_spec"]).read_text())
    # The existing environment specification contains only ASCII keys and values.
    spec_hash = canonical_hash(spec)
    for filename in ["environment_acceptance.json", "agent_acceptance.json"]:
        accepted = json.loads((ROOT / "reports/llm_extraction" / filename).read_text())
        if accepted["status"] != "passed" or accepted["spec_sha256"] != spec_hash:
            raise ValueError("The unchanged runtime lacks matching acceptance evidence.")
    if torch.__version__ != spec["torch"] or transformers.__version__ != spec["transformers"]:
        raise ValueError("Inference was launched in a different runtime.")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / spec["model_path"], local_files_only=True, trust_remote_code=False)
    prompts = [prepare_prompt(tokenizer, task, spec["maximum_context_tokens"], plan["generation"]["max_new_tokens"]) for task in tasks]
    preflight_report = {**manifest, "max_input_tokens": max(prompt["input_tokens"] for prompt in prompts),
                        "complete_evidence_count": sum(prompt["complete_evidence"] for prompt in prompts),
                        "dropped_fragments": sum(prompt["dropped_fragments"] for prompt in prompts),
                        "all_prompts_within_context": True, "hard_nonthinking": True}
    write_json(directory / "preflight.json", preflight_report)
    print(json.dumps({"event": "prompts_prepared", **preflight_report}), flush=True)
    if preflight:
        return
    if not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "2"}:
        raise ValueError("An explicitly authorized CUDA device is required.")
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(2)
    cache = directory / "prompts"
    cache.mkdir(exist_ok=True)
    model, records, failures = None, [], 0
    for number, (task, prompt) in enumerate(zip(tasks, prompts, strict=True)):
        identity = canonical_hash({"messages": prompt["messages"], "generation": plan["generation"],
                                   "model_manifest_sha256": spec["model_manifest_sha256"]})
        path = cache / f"{task['task_id']}.json"
        if path.exists():
            raw = json.loads(path.read_text())
            if raw["prompt_identity"] != identity or raw["task_id"] != task["task_id"]:
                raise ValueError("Cached prompt differs from the exact current input.")
            if hashlib.sha256(raw["response"].encode()).hexdigest() != raw["response_sha256"]:
                raise ValueError("Cached response is corrupted.")
        else:
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(ROOT / spec["model_path"], local_files_only=True, trust_remote_code=False,
                    torch_dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation=spec["attention"]).eval()
            inputs = tokenizer(prompt["text"], add_special_tokens=False, return_tensors="pt").to("cuda")
            step = time.time()
            with torch.inference_mode():
                generated = model.generate(**inputs, **plan["generation"], temperature=None, top_p=None, top_k=None,
                                           pad_token_id=tokenizer.eos_token_id)
            output_ids = generated[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            raw = {"task_id": task["task_id"], "prompt_identity": identity, "response": response,
                   "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                   "input_tokens": prompt["input_tokens"], "output_tokens": len(output_ids),
                   "hit_token_limit": len(output_ids) == plan["generation"]["max_new_tokens"], "elapsed_seconds": time.time()-step}
            atomic_json(path, raw)
        try:
            parsed = parse_decision(raw["response"], prompt["fragments"])
            parse_error = None
        except (ValueError, TypeError) as error:
            parsed, parse_error = {"decision": "unclear", "evidence": []}, str(error)
            failures += 1
        record = {**raw, **parsed, "parse_error": parse_error, "pmc_id": task["pmc_id"],
                  "patient_id": task["patient_id"], "concept": task["concept"],
                  "complete_evidence": prompt["complete_evidence"], "dropped_fragments": prompt["dropped_fragments"],
                  "retained_fragments": prompt["fragments"]}
        records.append(record)
        progress = {"event": "task_complete", "fold": fold, "completed": number+1, "total": len(tasks),
                    "format_errors": failures, "decisions": dict(Counter(item["decision"] for item in records)),
                    "elapsed_seconds": time.time()-started}
        atomic_json(directory / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    write_json(directory / "decisions.json", records)
    summary = {**manifest, **preflight_report, "status": "completed", "decisions_sha256": digest_file(directory / "decisions.json"),
               "format_success_rate": 1-failures/len(tasks), "format_errors": failures,
               "elapsed_seconds": time.time()-started, "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
               "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
               "generated_tokens": sum(record["output_tokens"] for record in records),
               "generation_seconds": sum(record["elapsed_seconds"] for record in records),
               "decisions": dict(Counter(record["decision"] for record in records))}
    write_json(directory / "summary.json", summary)
    print(json.dumps({"event": "completed", **summary}), flush=True)


def evaluate_pilot(plan_path, plan):
    from patientphex.evaluation import evaluate
    from patientphex.ontology import Ontology
    from scripts.optimize_cpu_association import bootstrap_delta, error_counts, validate_file

    report = ROOT / plan["report_dir"]
    if (report / "summary.json").exists():
        raise FileExistsError("Do not overwrite a completed association-review evaluation.")
    lookup = {str(document["pmc_id"]): document for document in read_jsonl(ROOT / plan["train_path"])}
    baseline = {str(record["pmc_id"]): record for record in read_jsonl(ROOT / plan["baseline_oof"])}
    folds = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    documents, original, rows, inference = [], [], [], []
    predictions = {policy: [] for policy in plan["policies"]}
    for fold in plan["pilot_folds"]:
        directory, manifest, tasks = load_tasks(plan_path, plan, fold)
        summary = json.loads((directory / "summary.json").read_text())
        if summary["status"] != "completed" or summary["plan_sha256"] != digest_file(plan_path):
            raise ValueError("Association inference is incomplete or belongs to another plan.")
        if summary["decisions_sha256"] != digest_file(directory / "decisions.json"):
            raise ValueError("Association decisions were altered.")
        records = json.loads((directory / "decisions.json").read_text())
        indexed = {record["task_id"]: record for record in records}
        if len(indexed) != len(records) or set(indexed) != {task["task_id"] for task in tasks}:
            raise ValueError("Decision/task coverage differs.")
        for task in tasks:
            record = indexed[task["task_id"]]
            if any(record[key] != task[key] for key in ["pmc_id", "patient_id", "concept"]):
                raise ValueError("A decision was attached to a different patient/concept.")
            if record["response_sha256"] != hashlib.sha256(record["response"].encode()).hexdigest():
                raise ValueError("Decision response checksum failed.")
            remaining = list(task["fragments"])
            for fragment in record["retained_fragments"]:
                remaining.remove(fragment)
            if record["complete_evidence"] != (task["all_occurrences_selected"] and not remaining):
                raise ValueError("Evidence coverage flag differs from the actual source excerpts.")
            try:
                parsed = parse_decision(record["response"], record["retained_fragments"])
                failed = False
            except (ValueError, TypeError):
                parsed, failed = {"decision": "unclear", "evidence": []}, True
            if parsed != {key: record[key] for key in ["decision", "evidence"]} or failed != bool(record["parse_error"]):
                raise ValueError("Decision parsing did not replay exactly.")
        failures = sum(bool(record["parse_error"]) for record in records)
        if failures != summary["format_errors"] or summary["format_success_rate"] != 1-failures/len(records):
            raise ValueError("Format failure summary differs from actual decisions.")
        validation = [lookup[identifier] for identifier in folds[fold]]
        local_base = [baseline[identifier] for identifier in folds[fold]]
        documents.extend(validation)
        original.extend(local_base)
        row = {"fold": fold, "baseline": evaluate(validation, local_base), "policies": {}}
        for policy in plan["policies"]:
            vetoes = {(record["pmc_id"], record["patient_id"], record["concept"]) for record in records
                      if should_veto(record["decision"], record["complete_evidence"], policy)}
            local = copy.deepcopy(local_base)
            removed = 0
            for record in local:
                for relation in record["association"]:
                    before = relation["phenotype"]
                    relation["phenotype"] = [concept for concept in before if (str(record["pmc_id"]), relation["patient_id"], concept) not in vetoes]
                    removed += len(before)-len(relation["phenotype"])
            if removed != len(vetoes) or any(a["entities"] != b["entities"] for a, b in zip(local, local_base, strict=True)):
                raise ValueError("Relation filtering edited entities or referenced nonexisting relations.")
            metrics = evaluate(validation, local)
            row["policies"][policy] = {"metrics": metrics, "removed_relations": removed}
            predictions[policy].extend(local)
            print(json.dumps({"fold": fold, "policy": policy, "score": metrics["score"],
                              "gain": metrics["score"]-row["baseline"]["score"], "removed_relations": removed}), flush=True)
        rows.append(row)
        inference.append(summary)
        write_json(report / f"fold{fold}_metrics.json", row)
    base = evaluate(documents, original)
    ranked, gates = [], plan["promotion"]
    for policy in plan["policies"]:
        metrics = evaluate(documents, predictions[policy])
        deltas = [row["policies"][policy]["metrics"]["score"]-row["baseline"]["score"] for row in rows]
        mean_gain = sum(deltas)/len(deltas)
        checks = {"minimum_mean_gain": mean_gain >= gates["minimum_mean_gain"],
                  "worst_fold": min(deltas) >= -gates["maximum_fold_loss"],
                  "entity_metrics_unchanged": all(metrics[key] == base[key] for key in ["mention", "document"]),
                  "association_guard": all(metrics[key]["f1"] >= base[key]["f1"]-gates["maximum_component_loss"]
                                           for key in ["association_micro", "association_macro"]),
                  "format_guard": all(item["format_success_rate"] >= gates["minimum_format_success_rate"] for item in inference)}
        ranked.append({"policy": policy, "metrics": metrics, "fold_deltas": deltas, "mean_gain": mean_gain,
                       "checks": checks, "eligible": all(checks.values()), "errors": error_counts(documents, predictions[policy])})
        write_jsonl(report / f"{policy}_predictions.jsonl", predictions[policy])
    ranked.sort(key=lambda item: (-item["mean_gain"], item["policy"]))
    eligible = [row for row in ranked if row["eligible"]]
    best = eligible[0] if eligible else ranked[0]
    validate_file(report / f"{best['policy']}_predictions.jsonl", documents, Ontology(ROOT / plan["ontology_path"]), report / "validation.json")
    result = {"plan_sha256": digest_file(plan_path), "baseline": base, "baseline_errors": error_counts(documents, original),
              "ranked": ranked, "selected": best["policy"] if eligible else None, "inference": inference,
              "best_bootstrap": bootstrap_delta(documents, original, predictions[best["policy"]]),
              "scope": "Two reused development folds; no B labels, no official score claim. Remaining-fold confirmation is mandatory before promotion."}
    write_json(report / "plan.json", plan)
    write_json(report / "summary.json", result)
    print(json.dumps({"selected": result["selected"], "best_mean_gain": best["mean_gain"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "preflight", "infer", "evaluate"])
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/llm_association/plan.json")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.stage == "prepare":
        prepare(args.plan, plan)
    elif args.stage == "evaluate":
        evaluate_pilot(args.plan, plan)
    else:
        infer(args.plan, plan, args.fold, preflight=args.stage == "preflight")
