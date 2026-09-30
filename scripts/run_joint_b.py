"""Prepare blind B shards and run the confirmed concept/association pipeline."""

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.abbreviations import extract_definitions
from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.llm_concept_review import build_task, eligible, ontology_definitions
from patientphex.ontology import Ontology
from patientphex.patient_linking import _check_provenance
from patientphex.span_linking import SpanLinker
from scripts.optimize_cpu_association import blind, from_scores
from scripts.run_llm_association import atomic_json, load_plan


def balanced_shards(tasks_by_document):
    """Keep documents intact, including zero-task documents, without using labels."""
    shards, loads = [[], []], [0, 0]
    for identifier in sorted(tasks_by_document, key=lambda i: (-len(tasks_by_document[i]), i)):
        shard = min(range(2), key=lambda i: (loads[i], len(shards[i]), i))
        shards[shard].append(identifier)
        loads[shard] += len(tasks_by_document[identifier])
    return [sorted(values) for values in shards]


def check_confirmation(plan):
    confirmation_plan = load_plan(ROOT / plan["confirmation_plan"])
    result = json.loads((ROOT / plan["confirmation_summary"]).read_text())
    expected_checks = {"remaining_mean_gain", "full_score_gain", "nonworse_folds", "component_guard",
                       "mention_improved", "format_guard", "positive_bootstrap_lower_bound"}
    if (result["plan_sha256"] != digest_file(ROOT / plan["confirmation_plan"])
            or result["selected"] != "joint_all_reviewed" or result["promoted"] is not True
            or set(result["checks"]) != expected_checks or not all(v is True for v in result["checks"].values())
            or plan["fixed"] != confirmation_plan["fixed"]):
        raise ValueError("The exact frozen joint confirmation must pass before any B tasks are prepared.")
    return confirmation_plan


def prepare(plan_path, plan):
    confirmation = check_confirmation(plan)
    concept_path = ROOT / plan["concept_plan"]
    concept = load_plan(concept_path)
    original = load_plan(ROOT / confirmation["concept_confirmation_plan"])
    for key in ["protocol_module", "generation", "fixed", "identity_fields", "environment_spec", "seed"]:
        if concept[key] != original[key]:
            raise ValueError("The B concept protocol differs from confirmed development inference.")
    if concept["pilot_folds"] != [0, 1] or concept["partition_role"] != "blind_b_shard":
        raise ValueError("Expected two document-disjoint B inference shards, not validation folds.")
    root = ROOT / concept["work_dir"]
    if root.exists():
        raise FileExistsError("Preserve earlier blind B concept tasks.")
    target = read_jsonl(ROOT / plan["target_path"])
    baseline = read_jsonl(ROOT / plan["baseline_b"])
    documents = {str(d["pmc_id"]): d for d in target}
    indexed = {str(d["pmc_id"]): d for d in baseline}
    train_ids = {str(d["pmc_id"]) for d in read_jsonl(ROOT / plan["train_path"])}
    if (len(target) != 100 or len(documents) != 100 or len(indexed) != len(baseline)
            or set(indexed) != set(documents) or train_ids & set(documents)
            or sum(len(d["patient"]) for d in target) != 244
            or any(d.get("entities") or d.get("association") for d in target)):
        raise ValueError("Expected the exact complete blind B source and matching baseline.")
    with (ROOT / plan["association_model"]).open("rb") as handle:
        structural = pickle.load(handle)
    _check_provenance(structural.provenance_, train_ids, set(documents))
    ontology = Ontology(ROOT / plan["ontology_path"])
    fixed, definitions = SpanLinker(ontology), ontology_definitions(ROOT / plan["ontology_path"])
    tasks_by_document = {}
    reproduced = []
    for source in target:
        identifier = str(source["pmc_id"])
        document, entities = blind(source), indexed[identifier]["entities"]
        reproduced.append(from_scores(document, entities, structural.predict_scores(document, entities), 0.5))
        local_definitions = extract_definitions(document)
        tasks_by_document[identifier] = [build_task(document, entity, i, ontology, definitions, local_definitions)
                                         for i, entity in enumerate(entities) if eligible(entity, ontology, fixed)]
    if reproduced != baseline:
        raise ValueError("The original B context baseline failed exact replay.")
    tasks = [task for values in tasks_by_document.values() for task in values]
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate blind B review identities.")
    for shard, identifiers in enumerate(balanced_shards(tasks_by_document)):
        values = [t for i in identifiers for t in tasks_by_document[i]]
        pack = {"fold": shard, "partition_role": "blind_b_shard", "tasks": values,
                "validation_document_ids": identifiers, "upstream_training_document_ids": sorted(train_ids),
                "llm_training_document_ids": [], "demonstration_document_ids": [],
                "answers_removed_before_prompting": True, "baseline_exact_replay": True,
                "plan_sha256": digest_file(concept_path), "parent_plan_sha256": digest_file(plan_path)}
        directory = root / f"fold{shard}"
        write_json(directory / "tasks.json", pack)
        manifest = {k: v for k, v in pack.items() if k != "tasks"}
        manifest.update(task_count=len(values), tasks_sha256=digest_file(directory / "tasks.json"))
        write_json(directory / "prepared.json", manifest)
        print(json.dumps({"shard": shard, "documents": len(identifiers), "tasks": len(values)}), flush=True)


def queue(plan_path, plan):
    check_confirmation(plan)
    report = ROOT / plan["report_dir"]
    state_path = report / "queue.json"
    if state_path.exists():
        raise FileExistsError("Inspect the existing B queue before any restart.")
    report.mkdir(parents=True, exist_ok=True)
    started = time.time()
    state = {"status": "preparing", "pid": os.getpid(), "plan_sha256": digest_file(plan_path),
             "started_at_epoch": started, "jobs": []}
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    try:
        atomic_json(state_path, state)
        prepare(plan_path, plan)
        processes = []
        for shard, gpu in enumerate([2, 0]):
            log = report / f"shard{shard}_concept.log"
            with log.open("x", encoding="utf-8") as handle:
                process = subprocess.Popen(["bash", "scripts/run_concept_review.sh", "--plan", plan["concept_plan"], "--fold", str(shard)],
                    cwd=ROOT, env={**env, "LLM_GPU_INDEX": str(gpu)}, stdout=handle, stderr=subprocess.STDOUT)
            job = {"phase": "concept", "shard": shard, "gpu": gpu, "pid": process.pid, "status": "running"}
            state["jobs"].append(job)
            processes.append((process, job))
        state["status"] = "concept_review_running"
        while processes:
            for process, job in list(processes):
                code = process.poll()
                if code is not None:
                    job.update(status="completed" if code == 0 else "failed", exit_code=code)
                    processes.remove((process, job))
            atomic_json(state_path, state)
            if processes:
                time.sleep(10)
        if any(j["status"] != "completed" for j in state["jobs"]):
            raise RuntimeError("B concept inference failed; association inference was not launched.")
        with (report / "association.log").open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(["bash", "scripts/run_relation_b.sh", "--plan", str(plan_path)], cwd=ROOT,
                env={**env, "RELATION_GPU_INDEX": "2"}, stdout=handle, stderr=subprocess.STDOUT)
        job = {"phase": "association", "gpu": 2, "pid": process.pid, "status": "running"}
        state["jobs"].append(job)
        state["status"] = "association_running"
        atomic_json(state_path, state)
        code = process.wait()
        job.update(status="completed" if code == 0 else "failed", exit_code=code)
        if code != 0:
            raise RuntimeError("B association inference failed; all intermediate evidence is preserved.")
        summary = json.loads((report / "summary.json").read_text())
        if summary["status"] != "completed" or digest_file(ROOT / plan["output"]) != summary["sha256"]:
            raise ValueError("B output is incomplete or differs from the validated candidate.")
        state.update(status="completed", output=plan["output"], sha256=summary["sha256"])
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.time() - started
        atomic_json(state_path, state)
        concept = json.loads((ROOT / plan["concept_plan"]).read_text())
        for shard in concept["pilot_folds"]:
            for name in ["prepared.json", "preflight.json", "summary.json", "progress.json"]:
                source = ROOT / concept["work_dir"] / f"fold{shard}" / name
                if source.exists():
                    shutil.copyfile(source, report / f"shard{shard}_concept_{name}")
        write_json(report / "artifact_manifest.json", {p.relative_to(report).as_posix(): digest_file(p)
            for p in sorted(report.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    queue(args.plan.resolve(), load_plan(args.plan))
