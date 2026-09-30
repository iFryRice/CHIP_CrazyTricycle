"""Confirm one frozen text/structure blend on the remaining development folds."""

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


def confirm(plan_path, plan):
    report = ROOT / plan["report_dir"]
    if (report / "summary.json").exists():
        raise FileExistsError("Do not overwrite confirmation results.")
    pilot_plan = load_plan(ROOT / plan["parent_plan"])
    pilot_summary = json.loads((ROOT / plan["parent_summary"]).read_text())
    selected = plan["selected"]
    if pilot_summary["selected"] != selected or selected != {"name": "blend_quarter", "neural_weight": 0.25}:
        raise ValueError("Policy changed after the two-fold selection.")
    if plan["parameters"] != pilot_plan["parameters"] or plan["representation"] != pilot_plan["representation"]:
        raise ValueError("Training or text representation changed after selection.")
    documents = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["train_path"])}
    baseline = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["baseline_oof"])}
    folds = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    pilot_predictions = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["parent_predictions"])}
    if set(pilot_predictions) != set(sum([folds[f] for f in pilot_plan["pilot_folds"]], [])):
        raise ValueError("Pilot prediction coverage changed.")
    predictions, rows = [], []
    for fold, val_ids in enumerate(folds):
        is_pilot = fold in pilot_plan["pilot_folds"]
        work = ROOT / (pilot_plan["work_dir"] if is_pilot else plan["work_dir"]) / f"fold{fold}"
        expected_plan = digest_file(ROOT / plan["parent_plan"]) if is_pilot else digest_file(plan_path)
        manifest = json.loads((work / "manifest.json").read_text())
        summary = json.loads((work / "summary.json").read_text())
        if (manifest["plan_sha256"] != expected_plan or manifest["fold"] != fold or manifest["smoke"]
                or set(manifest["training_document_ids"]) != set(documents)-set(val_ids)
                or manifest["validation_document_ids"] != val_ids):
            raise ValueError("A text model crosses the held-out partition or differs from its frozen run.")
        if summary["status"] != "completed" or summary["epochs_completed"] != plan["parameters"]["epochs"] or summary["plan_sha256"] != expected_plan:
            raise ValueError("A model did not complete its fixed training budget.")
        for name, key in [("last.pt", "checkpoint_sha256"), ("validation_scores.json", "scores_sha256")]:
            if digest_file(work / name) != summary[key]:
                raise ValueError("Completed model or predictions changed.")
        scores = json.loads((work / "validation_scores.json").read_text())
        indexed = {(r["pmc_id"], r["patient_id"], r["concept"]): r["score"] for r in scores}
        if len(indexed) != len(scores) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in indexed.values()):
            raise ValueError("Duplicate or invalid model probabilities.")
        with (ROOT / plan["association_models"][str(fold)]).open("rb") as handle:
            structural = pickle.load(handle)
        _check_provenance(structural.provenance_, set(documents)-set(val_ids), set(val_ids))
        validation = [documents[i] for i in val_ids]
        base = [baseline[i] for i in val_ids]
        local, consumed = [], set()
        for document, original in zip(validation, base, strict=True):
            old = structural.predict_scores(blind(document), original["entities"])
            if from_scores(document, original["entities"], old, 0.5) != original:
                raise ValueError("Original fivefold baseline no longer replays exactly.")
            mixed = []
            for score in old:
                key = str(document["pmc_id"]), score["patient_id"], score["concept"]
                consumed.add(key)
                mixed.append({**score, "score": 0.75*score["score"]+0.25*indexed[key]})
            prediction = from_scores(document, original["entities"], mixed, 0.5)
            if is_pilot and prediction != pilot_predictions[str(document["pmc_id"])]:
                raise ValueError("The frozen blend failed exact pilot replay.")
            if prediction["entities"] != original["entities"]:
                raise ValueError("Entities changed in an association-only experiment.")
            local.append(prediction)
        if consumed != set(indexed):
            raise ValueError("Text and structural candidate coverage differs.")
        predictions.extend(local)
        base_metrics, metrics = evaluate(validation, base), evaluate(validation, local)
        row = {"fold": fold, "baseline": base_metrics, "metrics": metrics, "delta": metrics["score"]-base_metrics["score"],
               "text_checkpoint_sha256": summary["checkpoint_sha256"], "text_scores_sha256": summary["scores_sha256"]}
        rows.append(row)
        write_json(report / f"fold{fold}_metrics.json", row)
        print(json.dumps({"fold": fold, "score": metrics["score"], "gain": row["delta"]}), flush=True)
    ordered_source = list(documents.values())
    baseline_metrics = evaluate(ordered_source, list(baseline.values()))
    metrics = evaluate(ordered_source, predictions)
    remaining_mean = sum(rows[f]["delta"] for f in plan["pilot_folds"])/len(plan["pilot_folds"])
    bootstrap = bootstrap_delta(ordered_source, list(baseline.values()), predictions)
    gates = plan["confirmation_requirement"]
    checks = {"remaining_mean_gain": remaining_mean >= gates["minimum_remaining_mean_gain"],
              "full_score_gain": metrics["score"]-baseline_metrics["score"] >= gates["minimum_full_score_gain"],
              "nonworse_folds": sum(row["delta"] >= 0 for row in rows) >= gates["minimum_nonworse_folds"],
              "component_guard": all(metrics[k]["f1"] >= baseline_metrics[k]["f1"]-gates["maximum_component_f1_loss"] for k in ["mention","document","association_micro","association_macro"]),
              "entities_unchanged": all(metrics[k] == baseline_metrics[k] for k in ["mention", "document"]),
              "positive_bootstrap_lower_bound": bootstrap["interval_95"][0] > 0}
    write_jsonl(report / "development_oof.jsonl", predictions)
    validation = validate_file(report / "development_oof.jsonl", ordered_source, Ontology(ROOT / plan["ontology_path"]), report / "validation.json")
    result = {"plan_sha256": digest_file(plan_path), "selected": selected, "promoted": all(checks.values()),
              "baseline": baseline_metrics, "metrics": metrics, "fold_deltas": [row["delta"] for row in rows],
              "confirmation_mean_gain": remaining_mean, "checks": checks, "bootstrap": bootstrap,
              "errors": error_counts(ordered_source, predictions), "validation": validation,
              "scope": "Fivefold repeated development; policy frozen after folds 0/1. Previously used documents, not independent test or official B accuracy."}
    write_json(report / "summary.json", result)
    return result


def archive(plan_path, plan, state):
    report = ROOT / plan["report_dir"]
    lines = ["# 监督式关联模型：剩余三折确认", "", f"计划 SHA256：`{digest_file(plan_path)}`。", "",
             "固定使用 75% 原上下文模型 + 25% 监督文本模型，阈值 0.5。训练预算、文本表示和实体完全不变。", "",
             f"队列状态：`{state['status']}`。", ""]
    if (report / "summary.json").exists():
        result = json.loads((report / "summary.json").read_text())
        lines.extend([f"五折总分：{result['baseline']['score']:.6f} → {result['metrics']['score']:.6f}。通过确认：{result['promoted']}。", "",
                      f"剩余三折平均增益：{result['confirmation_mean_gain']:+.6f}；全折增益：`{result['fold_deltas']}`。", "",
                      f"预设门槛检查：`{result['checks']}`。", "", f"描述性配对 bootstrap 95% 区间：`{result['bootstrap']['interval_95']}`。"])
    if "error" in state:
        lines.extend(["", f"错误：`{state['error']}`。"])
    lines.extend(["", "门槛沿用两折试验前预注册的要求：剩余三折平均增益 ≥0.003，全五折增益 ≥0.005，至少四折不下降，各项 F1 损失 ≤0.002，bootstrap 下界 >0。",
                  "", "仅通过全部门槛才允许生成新的 B 候选。本报告为重复使用开发数据的结果，不是官方 B 分数；当前官方只确认纯 CPU 为 0.6152。", ""])
    (report / "report.md").write_text("\n".join(lines), encoding="utf-8")
    for fold in plan["pilot_folds"]:
        work = ROOT / plan["work_dir"] / f"fold{fold}"
        for name in ["manifest.json", "summary.json", "training.jsonl"]:
            if (work / name).exists(): shutil.copyfile(work / name, report / f"fold{fold}_{name}")
    write_json(report / "artifact_manifest.json", {p.relative_to(report).as_posix(): digest_file(p)
                for p in sorted(report.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"})


def queue(plan_path, plan):
    parent = load_plan(ROOT / plan["parent_plan"])
    summary = json.loads((ROOT / plan["parent_summary"]).read_text())
    if summary["selected"] != plan["selected"] or plan["parameters"] != parent["parameters"] or plan["representation"] != parent["representation"]:
        raise ValueError("Confirmation changed the selected model or training protocol.")
    report = ROOT / plan["report_dir"]
    state_path = report / "queue.json"
    if state_path.exists(): raise FileExistsError("Inspect the existing confirmation queue before resuming.")
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "plan.json", plan)
    started = time.time()
    state = {"status": "starting", "pid": os.getpid(), "plan_sha256": digest_file(plan_path), "jobs": [], "started_at_epoch": started}
    pending, active, failed = list(plan["pilot_folds"]), {}, False
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    try:
        while pending or active:
            for gpu in [2, 0]:
                if gpu in active or not pending or failed: continue
                fold = pending.pop(0)
                log = report / f"fold{fold}_training.log"
                with log.open("x", encoding="utf-8") as handle:
                    process = subprocess.Popen(["bash", "scripts/run_relation_text.sh", "--plan", str(plan_path), "--fold", str(fold)],
                              cwd=ROOT, env={**env, "RELATION_GPU_INDEX": str(gpu)}, stdout=handle, stderr=subprocess.STDOUT)
                job = {"fold": fold, "gpu": gpu, "pid": process.pid, "status": "running", "log": str(log.relative_to(ROOT))}
                state["jobs"].append(job); active[gpu] = process, job
            state["status"] = "training"
            for gpu, (process, job) in list(active.items()):
                code = process.poll()
                if code is not None:
                    job.update(status="completed" if code == 0 else "failed", exit_code=code)
                    failed |= code != 0
                    del active[gpu]
            atomic_json(state_path, state)
            if failed and not active: raise RuntimeError("Confirmation worker failed; pending folds not launched.")
            if pending or active: time.sleep(10)
        state["status"] = "evaluating"
        atomic_json(state_path, state)
        result = confirm(plan_path, plan)
        state.update(status="completed", promoted=result["promoted"], score=result["metrics"]["score"])
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.time()-started
        atomic_json(state_path, state)
        archive(plan_path, plan, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["queue", "evaluate"])
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.stage == "queue": queue(args.plan.resolve(), plan)
    else: confirm(args.plan.resolve(), plan)
