"""Queue, evaluate and report the fixed supervised relation pilot."""

import argparse
from datetime import datetime, timezone
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


def evaluate_pilot(plan_path, plan):
    report = ROOT / plan["report_dir"]
    if (report / "summary.json").exists():
        raise FileExistsError("Do not overwrite completed evaluation.")
    documents = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["train_path"])}
    baseline = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["baseline_oof"])}
    folds = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    source, original, fold_metrics = [], [], []
    outputs = {config["name"]: [] for config in plan["configurations"]}
    for fold in plan["pilot_folds"]:
        work = ROOT / plan["work_dir"] / f"fold{fold}"
        manifest = json.loads((work / "manifest.json").read_text())
        summary = json.loads((work / "summary.json").read_text())
        val_ids = folds[fold]
        if (manifest["smoke"] or manifest["fold"] != fold or manifest["plan_sha256"] != digest_file(plan_path)
                or set(manifest["training_document_ids"]) != set(documents)-set(val_ids)
                or manifest["validation_document_ids"] != val_ids):
            raise ValueError("Training provenance mismatch.")
        if summary["status"] != "completed" or summary["epochs_completed"] != plan["parameters"]["epochs"] or summary["plan_sha256"] != digest_file(plan_path):
            raise ValueError("Training did not finish its exact frozen budget.")
        for name, key in [("last.pt", "checkpoint_sha256"), ("validation_scores.json", "scores_sha256")]:
            if digest_file(work / name) != summary[key]:
                raise ValueError("Model or predictions changed after inference.")
        raw_scores = json.loads((work / "validation_scores.json").read_text())
        if any(not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1 for row in raw_scores):
            raise ValueError("Invalid relation probability.")
        scored = {(r["pmc_id"], r["patient_id"], r["concept"]): r["score"] for r in raw_scores}
        raw_examples = json.loads((work / "validation_examples.json").read_text())
        if digest_file(work / "validation_examples.json") != manifest["validation_examples_sha256"]:
            raise ValueError("Validation examples changed after preparation.")
        if len(scored) != len(raw_scores) or set(scored) != {(r["pmc_id"], r["patient_id"], r["concept"]) for r in raw_examples}:
            raise ValueError("Prediction coverage differs from blind validation examples.")
        with (ROOT / plan["association_models"][str(fold)]).open("rb") as handle:
            context = pickle.load(handle)
        _check_provenance(context.provenance_, set(documents)-set(val_ids), set(val_ids))
        validation = [documents[i] for i in val_ids]
        base = [baseline[i] for i in val_ids]
        predictions = {key: [] for key in outputs}
        seen = set()
        for d, b in zip(validation, base, strict=True):
            old = context.predict_scores(blind(d), b["entities"])
            if from_scores(d, b["entities"], old, 0.5) != b:
                raise ValueError("The existing context baseline failed exact replay.")
            for config in plan["configurations"]:
                alpha = config["neural_weight"]
                scores = []
                for row in old:
                    key = str(d["pmc_id"]), row["patient_id"], row["concept"]
                    seen.add(key)
                    scores.append({**row, "score": (1-alpha)*row["score"]+alpha*scored[key]})
                prediction = from_scores(d, b["entities"], scores, 0.5)
                if prediction["entities"] != b["entities"]:
                    raise ValueError("Relation-only experiment changed entities.")
                predictions[config["name"]].append(prediction)
        if seen != set(scored):
            raise ValueError("Text and structural candidate universes differ.")
        row = {"fold": fold, "baseline": evaluate(validation, base), "configurations": {}}
        for name, records in predictions.items():
            row["configurations"][name] = evaluate(validation, records)
            outputs[name].extend(records)
        source.extend(validation); original.extend(base); fold_metrics.append(row)
        write_json(report / f"fold{fold}_metrics.json", row)
        for name in ["manifest.json", "summary.json"]:
            shutil.copyfile(work / name, report / f"fold{fold}_{name}")
        print(json.dumps({"fold": fold, "scores": {k: v["score"] for k,v in row["configurations"].items()}}), flush=True)
    base_metrics = evaluate(source, original)
    ranked = []
    gates = plan["promotion"]
    for config in plan["configurations"]:
        name = config["name"]
        metrics = evaluate(source, outputs[name])
        deltas = [row["configurations"][name]["score"]-row["baseline"]["score"] for row in fold_metrics]
        checks = {"minimum_mean_gain": sum(deltas)/len(deltas) >= gates["minimum_mean_gain"],
                  "worst_fold": min(deltas) >= -gates["maximum_fold_loss"],
                  "association_guard": all(metrics[k]["f1"] >= base_metrics[k]["f1"]-gates["maximum_component_loss"] for k in ["association_micro", "association_macro"]),
                  "entities_unchanged": all(metrics[k] == base_metrics[k] for k in ["mention", "document"])}
        ranked.append({"configuration": config, "metrics": metrics, "fold_deltas": deltas,
                       "mean_gain": sum(deltas)/len(deltas), "checks": checks,
                       "eligible": all(checks.values()), "errors": error_counts(source, outputs[name])})
        write_jsonl(report / f"{name}_predictions.jsonl", outputs[name])
        validate_file(report / f"{name}_predictions.jsonl", source, Ontology(ROOT / plan["ontology_path"]), report / f"{name}_validation.json")
    ranked.sort(key=lambda row: (-row["mean_gain"], row["configuration"]["name"]))
    passing = [row for row in ranked if row["eligible"]]
    best = (passing or ranked)[0]
    result = {"plan_sha256": digest_file(plan_path), "baseline": base_metrics, "ranked": ranked,
              "selected": best["configuration"] if passing else None,
              "bootstrap": bootstrap_delta(source, original, outputs[best["configuration"]["name"]]),
              "scope": "Two reused development folds. The classifier and upstream candidates exclude each held-out fold. Not independent test or official B accuracy."}
    write_json(report / "summary.json", result)
    return result


def report_result(plan_path, plan, state):
    report = ROOT / plan["report_dir"]
    summary_path = report / "summary.json"
    lines = ["# 监督式患者–表型文本分类", "", f"计划 SHA256：`{digest_file(plan_path)}`。", "",
             "现有 BiomedBERT 完整微调，固定 3 个 epoch；每折训练候选来自仅使用该折训练文献的内层交叉预测。目标患者、其他患者、表型均显式标记。实体和 HPO 映射保持固定。", "",
             f"队列状态：`{state['status']}`。官方 B 反馈仍仅确认纯 CPU 文件为 0.6152。", ""]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        lines.extend(["| 方案 | 两折汇总分 | 平均折增益 | 晋级 |", "|---|---:|---:|---|"])
        lines.append(f"| 上下文基线 | {summary['baseline']['score']:.6f} | — | — |")
        for row in summary["ranked"]:
            lines.append(f"| {row['configuration']['name']} | {row['metrics']['score']:.6f} | {row['mean_gain']:+.6f} | {'是' if row['eligible'] else '否'} |")
        lines.extend(["", f"选择：`{summary['selected']}`。通过后还需固定策略验证其余三折，未生成或提交 B 文件。"])
    if "error" in state:
        lines.extend(["", f"运行错误：`{state['error']}`。已保存日志，请先定位原因。"])
    lines.extend(["", "两折预设门槛：平均增益 ≥0.008，最差折损失 ≤0.002，关联微/宏 F1 损失 ≤0.002，实体指标不变。",
                  "", "文献用于过往开发和本次模型选择，不能视为独立测试。文本按固定分片 token 预算裁剪，保留患者或实体焦点；完整原文片段与裁剪数量记录在服务器工作目录。",
                  "", f"模型与原始训练例：`{plan['work_dir']}`；运行状态：`queue.json`；训练数值：`fold*_training.jsonl`。",
                  "", "全部服务器文件实际位于 `/home/dcf/chip2026/`；复用已验收环境，没有修改依赖或其他用户进程。", ""])
    (report / "report.md").write_text("\n".join(lines), encoding="utf-8")
    for fold in plan["pilot_folds"]:
        work = ROOT / plan["work_dir"] / f"fold{fold}"
        for name in ["manifest.json", "summary.json", "training.jsonl"]:
            if (work / name).exists():
                shutil.copyfile(work / name, report / f"fold{fold}_{name}")
    write_json(report / "artifact_manifest.json", {p.relative_to(report).as_posix(): digest_file(p)
                for p in sorted(report.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"})


def queue(plan_path, plan):
    report = ROOT / plan["report_dir"]
    state_path = report / "queue.json"
    if state_path.exists():
        raise FileExistsError("Check the current queue instead of restarting it.")
    smoke = json.loads((ROOT / plan["work_dir"] / "smoke/summary.json").read_text())
    if smoke["status"] != "smoke_complete" or smoke["plan_sha256"] != digest_file(plan_path) or smoke["optimizer_steps"] < plan["smoke_steps"]-3:
        raise ValueError("The fixed-plan smoke gate has not passed.")
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "plan.json", plan)
    started = time.time()
    state = {"status": "starting", "pid": os.getpid(), "plan_sha256": digest_file(plan_path), "jobs": [], "started_at_epoch": started}
    pending, active, failed = list(plan["pilot_folds"]), {}, False
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    try:
        while pending or active:
            for gpu in [2, 0]:
                if gpu in active or not pending or failed:
                    continue
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
            if failed and not active:
                raise RuntimeError("A training worker failed; no pending folds launched.")
            if pending or active:
                time.sleep(10)
        state["status"] = "evaluating"
        atomic_json(state_path, state)
        result = evaluate_pilot(plan_path, plan)
        state.update(status="completed", selected=result["selected"], best_mean_gain=result["ranked"][0]["mean_gain"])
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.time()-started
        atomic_json(state_path, state)
        report_result(plan_path, plan, state)


def status(plan):
    report = ROOT / plan["report_dir"]
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "folds": []}
    if (report / "queue.json").exists():
        result["queue"] = json.loads((report / "queue.json").read_text())
    for fold in plan["pilot_folds"]:
        work = ROOT / plan["work_dir"] / f"fold{fold}"
        record = {"fold": fold}
        if (work / "training.jsonl").exists():
            rows = [json.loads(line) for line in (work / "training.jsonl").read_text().splitlines()]
            record["latest"] = rows[-1]
            steps = [row for row in rows if row["event"] == "train_step"]
            if len(steps) > 1 and rows[-1]["event"] != "complete":
                first, last = steps[max(0,len(steps)-4)], steps[-1]
                seconds = (last["elapsed_seconds"]-first["elapsed_seconds"])/max(1, last["step"]-first["step"])
                record["estimated_training_seconds_remaining"] = seconds*(last["planned_steps"]-last["step"])
        result["folds"].append(record)
    if (report / "summary.json").exists():
        summary = json.loads((report / "summary.json").read_text())
        result["evaluation"] = {"baseline_score": summary["baseline"]["score"], "selected": summary["selected"],
                                "ranked": [{k:v for k,v in r.items() if k != "metrics"} | {"score":r["metrics"]["score"]} for r in summary["ranked"]]}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["queue", "evaluate", "status"])
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text()) if args.stage == "status" else load_plan(args.plan)
    if args.stage == "queue": queue(args.plan.resolve(), plan)
    elif args.stage == "evaluate": evaluate_pilot(args.plan.resolve(), plan)
    else: status(plan)
