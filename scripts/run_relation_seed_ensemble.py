"""Test one predeclared three-seed relation ensemble without selecting seeds."""

import argparse
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.patient_linking import _check_provenance
from scripts.optimize_cpu_association import blind, bootstrap_delta, error_counts, from_scores, validate_file
from scripts.run_llm_association import atomic_json, load_plan


def verified_scores(member, fold, documents, val_ids, parameters):
    work = ROOT / member["work_dir"] / f"fold{fold}"
    manifest = json.loads((work / "manifest.json").read_text())
    summary = json.loads((work / "summary.json").read_text())
    expected = member["fold_plan_sha256"][str(fold)]
    expected_parameters = {**parameters, "seed": member["seed"]}
    if (manifest["plan_sha256"] != expected or manifest["smoke"] or manifest["fold"] != fold
            or manifest["parameters"] != expected_parameters
            or set(manifest["training_document_ids"]) != set(documents)-set(val_ids)
            or manifest["validation_document_ids"] != val_ids):
        raise ValueError("Ensemble member differs from the fixed seed, training budget or partition.")
    if summary["status"] != "completed" or summary["epochs_completed"] != parameters["epochs"] or summary["plan_sha256"] != expected:
        raise ValueError("An ensemble member did not complete the fixed budget.")
    if digest_file(work / "last.pt") != summary["checkpoint_sha256"] or digest_file(work / "validation_scores.json") != summary["scores_sha256"]:
        raise ValueError("Completed model or scores changed.")
    rows = json.loads((work / "validation_scores.json").read_text())
    indexed = {(row["pmc_id"], row["patient_id"], row["concept"]): row["score"] for row in rows}
    if len(indexed) != len(rows) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in indexed.values()):
        raise ValueError("Invalid or duplicate ensemble probabilities.")
    return indexed, {"checkpoint_sha256": summary["checkpoint_sha256"], "scores_sha256": summary["scores_sha256"]}


def evaluate_ensemble(plan_path, plan):
    report = ROOT / plan["report_dir"]
    if (report / "summary.json").exists(): raise FileExistsError("Preserve previous seed-ensemble evaluation.")
    documents = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["train_path"])}
    baseline = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["baseline_oof"])}
    folds = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    predictions, rows, artifacts = [], [], {}
    for fold, val_ids in enumerate(folds):
        members = []
        for seed in plan["seeds"]:
            member = dict(plan["members"][str(seed)])
            if "fold_work_dirs" in member: member["work_dir"] = member["fold_work_dirs"][str(fold)]
            scores, evidence = verified_scores(member, fold, documents, val_ids, plan["parameters"])
            members.append(scores)
            artifacts[f"seed{seed}_fold{fold}"] = evidence
        keys = set(members[0])
        if any(set(member) != keys for member in members):
            raise ValueError("Seed members have different candidate coverage.")
        averaged = {key: sum(member[key] for member in members)/len(members) for key in keys}
        with (ROOT / plan["association_models"][str(fold)]).open("rb") as handle:
            structural = pickle.load(handle)
        _check_provenance(structural.provenance_, set(documents)-set(val_ids), set(val_ids))
        validation = [documents[i] for i in val_ids]
        base = [baseline[i] for i in val_ids]
        local, consumed = [], set()
        for document, old in zip(validation, base, strict=True):
            scores = structural.predict_scores(blind(document), old["entities"])
            if from_scores(document, old["entities"], scores, 0.5) != old:
                raise ValueError("Unchanged structural baseline failed exact replay.")
            combined = []
            for score in scores:
                key = str(document["pmc_id"]), score["patient_id"], score["concept"]
                consumed.add(key)
                combined.append({**score, "score": 0.75*score["score"]+0.25*averaged[key]})
            prediction = from_scores(document, old["entities"], combined, 0.5)
            if prediction["entities"] != old["entities"]: raise ValueError("An association experiment changed entities.")
            local.append(prediction)
        if consumed != keys: raise ValueError("Text and structural candidate universes differ.")
        metrics, base_metrics = evaluate(validation, local), evaluate(validation, base)
        row = {"fold": fold, "baseline": base_metrics, "metrics": metrics, "delta": metrics["score"]-base_metrics["score"]}
        rows.append(row); predictions.extend(local)
        write_json(report / f"fold{fold}_metrics.json", row)
        print(json.dumps({"fold": fold, "score": metrics["score"], "gain": row["delta"]}), flush=True)
    source, original = list(documents.values()), list(baseline.values())
    metrics, base_metrics = evaluate(source, predictions), evaluate(source, original)
    remaining = sum(rows[f]["delta"] for f in [2,3,4])/3
    bootstrap = bootstrap_delta(source, original, predictions)
    gates = plan["confirmation_requirement"]
    checks = {"remaining_mean_gain": remaining >= gates["minimum_remaining_mean_gain"],
              "full_score_gain": metrics["score"]-base_metrics["score"] >= gates["minimum_full_score_gain"],
              "nonworse_folds": sum(row["delta"] >= 0 for row in rows) >= gates["minimum_nonworse_folds"],
              "component_guard": all(metrics[k]["f1"] >= base_metrics[k]["f1"]-gates["maximum_component_f1_loss"] for k in ["mention","document","association_micro","association_macro"]),
              "entities_unchanged": all(metrics[k] == base_metrics[k] for k in ["mention", "document"]),
              "positive_bootstrap_lower_bound": bootstrap["interval_95"][0] > 0}
    write_jsonl(report / "development_oof.jsonl", predictions)
    validation = validate_file(report / "development_oof.jsonl", source, Ontology(ROOT / plan["ontology_path"]), report / "validation.json")
    result = {"plan_sha256": digest_file(plan_path), "selected": plan["selected"], "seeds": plan["seeds"],
              "promoted": all(checks.values()), "baseline": base_metrics, "metrics": metrics,
              "fold_deltas": [row["delta"] for row in rows], "confirmation_mean_gain": remaining,
              "checks": checks, "bootstrap": bootstrap, "errors": error_counts(source, predictions),
              "artifacts": artifacts, "validation": validation,
              "scope": "Predeclared three-seed probability mean, no seed or threshold selection. Repeated fivefold development informed by prior errors; not independent confirmation or official B accuracy."}
    write_json(report / "summary.json", result)
    return result


def archive(plan_path, plan, state):
    report = ROOT / plan["report_dir"]
    lines = ["# 三随机种子关联融合", "", f"冻结计划 SHA256：`{digest_file(plan_path)}`。", "",
             "为检查单次训练波动，新增两个预先固定种子；每折对全部三个种子的概率等权平均，再按 25% 权重与原上下文模型融合。没有挑选最优种子或调整阈值。", "",
             f"状态：`{state['status']}`。", ""]
    if (report / "summary.json").exists():
        s = json.loads((report / "summary.json").read_text())
        lines.extend([f"五折开发总分：{s['baseline']['score']:.6f} → {s['metrics']['score']:.6f}；晋级：{s['promoted']}。", "",
                      f"逐折增益：`{s['fold_deltas']}`；第 2、3、4 折平均增益：{s['confirmation_mean_gain']:+.6f}。", "",
                      f"预设门槛：`{s['checks']}`；bootstrap 区间：`{s['bootstrap']['interval_95']}`。"])
    if "error" in state: lines.extend(["", f"错误：`{state['error']}`。"])
    lines.extend(["", "沿用原完整门槛，不因前一轮边缘结果放宽要求。此试验再次使用开发文献，不是新的独立验证，也没有官方 B 分数。", "",
                  "模型与完整日志实际保存在服务器 `/home/dcf/chip2026/`；仅使用已授权 GPU 0、2。未自动向比赛平台提交。", ""])
    (report / "report.md").write_text("\n".join(lines), encoding="utf-8")
    for seed in plan["new_seeds"]:
        for fold in range(5):
            work = ROOT / plan["members"][str(seed)]["work_dir"] / f"fold{fold}"
            for name in ["manifest.json", "summary.json", "training.jsonl"]:
                if (work / name).exists(): shutil.copyfile(work / name, report / f"seed{seed}_fold{fold}_{name}")
    write_json(report / "artifact_manifest.json", {p.relative_to(report).as_posix(): digest_file(p)
                for p in sorted(report.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"})


def queue(plan_path, plan):
    for seed in plan["new_seeds"]:
        child = load_plan(ROOT / plan["members"][str(seed)]["plan"])
        if child["parameters"] != {**plan["parameters"], "seed": seed} or child["representation"] != plan["representation"]:
            raise ValueError("A child training plan changed more than the declared seed.")
    report = ROOT / plan["report_dir"]
    state_path = report / "queue.json"
    if state_path.exists(): raise FileExistsError("Inspect the existing seed queue before retrying.")
    if shutil.disk_usage(ROOT).free < 25*2**30: raise RuntimeError("Insufficient reserved disk space for ten checkpoints.")
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "plan.json", plan)
    started = time.time()
    state = {"status": "starting", "pid": os.getpid(), "plan_sha256": digest_file(plan_path), "jobs": [], "started_at_epoch": started}
    pending = [(seed,fold) for seed in plan["new_seeds"] for fold in range(5)]
    active, failed = {}, False
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    try:
        while pending or active:
            for gpu in [2,0]:
                if gpu in active or not pending or failed: continue
                seed,fold = pending.pop(0)
                member = plan["members"][str(seed)]
                log = report / f"seed{seed}_fold{fold}.log"
                with log.open("x", encoding="utf-8") as handle:
                    process = subprocess.Popen(["bash", "scripts/run_relation_text.sh", "--plan", str(ROOT/member["plan"]), "--fold", str(fold)],
                              cwd=ROOT, env={**env,"RELATION_GPU_INDEX":str(gpu)}, stdout=handle, stderr=subprocess.STDOUT)
                job={"seed":seed,"fold":fold,"gpu":gpu,"pid":process.pid,"status":"running","log":str(log.relative_to(ROOT))}
                state["jobs"].append(job);active[gpu]=process,job
            state["status"]="training"
            state["pending_jobs"]=len(pending)
            for gpu,(process,job) in list(active.items()):
                code=process.poll()
                if code is not None:
                    job.update(status="completed" if code==0 else "failed",exit_code=code)
                    failed |= code!=0
                    del active[gpu]
            atomic_json(state_path,state)
            if failed and not active:raise RuntimeError("A seed worker failed; pending jobs were not launched.")
            if pending or active:time.sleep(10)
        state["status"]="evaluating";atomic_json(state_path,state)
        result=evaluate_ensemble(plan_path,plan)
        state.update(status="completed",promoted=result["promoted"],score=result["metrics"]["score"])
    except Exception as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"]=time.time()-started
        atomic_json(state_path,state)
        archive(plan_path,plan,state)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["queue","evaluate"])
    parser.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args()
    plan=load_plan(args.plan)
    if args.stage=="queue":queue(args.plan.resolve(),plan)
    else:evaluate_ensemble(args.plan.resolve(),plan)
