"""Evaluate one predeclared no-addition ablation of the completed seed ensemble."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file, read_jsonl, write_json, write_jsonl
from patientphex.evaluation import evaluate
from patientphex.ontology import Ontology
from patientphex.relation_veto import filter_existing
from scripts.optimize_cpu_association import bootstrap_delta, error_counts, validate_file
from scripts.run_llm_association import load_plan


def run(plan_path):
    plan = load_plan(plan_path)
    report = ROOT / plan["report_dir"]
    if report.exists(): raise FileExistsError("Do not overwrite a completed action-policy ablation.")
    source = read_jsonl(ROOT / plan["train_path"])
    documents = {str(d["pmc_id"]): d for d in source}
    baseline = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["baseline_oof"])}
    ensemble = {str(d["pmc_id"]): d for d in read_jsonl(ROOT / plan["ensemble_oof"])}
    folds = json.loads((ROOT / plan["split_path"]).read_text())["folds"]
    if (set(documents) != set(baseline) or set(documents) != set(ensemble)
            or sorted(sum(folds, [])) != sorted(documents)):
        raise ValueError("Source or prediction coverage differs from the frozen folds.")
    predictions = [filter_existing(baseline[str(d["pmc_id"])], ensemble[str(d["pmc_id"])]) for d in source]
    indexed = {str(d["pmc_id"]): d for d in predictions}
    rows = []
    for fold, identifiers in enumerate(folds):
        validation = [documents[i] for i in identifiers]
        old = evaluate(validation, [baseline[i] for i in identifiers])
        metrics = evaluate(validation, [indexed[i] for i in identifiers])
        rows.append({"fold": fold, "baseline": old, "metrics": metrics, "delta": metrics["score"]-old["score"]})
    old, metrics = evaluate(source, list(baseline.values())), evaluate(source, predictions)
    bootstrap = bootstrap_delta(source, list(baseline.values()), predictions)
    remaining = sum(rows[f]["delta"] for f in [2,3,4])/3
    gates = plan["confirmation_requirement"]
    checks = {"remaining_mean_gain": remaining >= gates["minimum_remaining_mean_gain"],
              "full_score_gain": metrics["score"]-old["score"] >= gates["minimum_full_score_gain"],
              "nonworse_folds": sum(row["delta"] >= 0 for row in rows) >= gates["minimum_nonworse_folds"],
              "component_guard": all(metrics[k]["f1"] >= old[k]["f1"]-gates["maximum_component_f1_loss"] for k in ["mention","document","association_micro","association_macro"]),
              "entities_unchanged": all(metrics[k] == old[k] for k in ["mention", "document"]),
              "positive_bootstrap_lower_bound": bootstrap["interval_95"][0] > 0}
    write_jsonl(report / "development_oof.jsonl", predictions)
    validate_file(report / "development_oof.jsonl", source, Ontology(ROOT / plan["ontology_path"]), report / "validation.json")
    for row in rows: write_json(report / f"fold{row['fold']}_metrics.json", row)
    result = {"plan_sha256": digest_file(plan_path), "selected": plan["selected"], "seeds": plan["seeds"],
              "promoted": all(checks.values()), "baseline": old, "metrics": metrics,
              "fold_deltas": [row["delta"] for row in rows], "confirmation_mean_gain": remaining,
              "checks": checks, "bootstrap": bootstrap, "errors": error_counts(source, predictions),
              "scope": "One action-policy ablation defined after prior development diagnostics. Models, blend and threshold fixed. Repeated development, not independent confirmation or official B accuracy."}
    write_json(report / "plan.json", plan);write_json(report / "summary.json", result)
    lines = ["# 监督式关联集成：只筛除关联的消融", "", f"计划 SHA256：`{digest_file(plan_path)}`。", "",
             "此前完整三种子融合仍在第 2 折新增较多错误关联。本次在评分前冻结一个策略：只保留原上下文模型与三种子融合都预测为阳性的关联，禁止新增。所有实体、HPO、模型、融合比例及阈值保持不变。", "",
             f"五折总分：{old['score']:.6f} → {metrics['score']:.6f}；通过全部门槛：{result['promoted']}。", "",
             f"逐折增益：`{result['fold_deltas']}`；后面三折平均增益：{remaining:+.6f}。", "",
             f"门槛检查：`{checks}`；bootstrap 区间：`{bootstrap['interval_95']}`。", "",
             "沿用原数值门槛，没有因上一轮失败放宽要求。该策略受到先前错误分析启发，使用同一批开发文献，因此不是独立验证，不能声称已达到官方 0.70。", "",
             "只有通过全部门槛才生成对应 B 候选；尚未向比赛平台提交。", ""]
    (report / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(report / "artifact_manifest.json", {p.name:digest_file(p) for p in sorted(report.iterdir()) if p.is_file() and p.name != "artifact_manifest.json"})
    print(json.dumps({"score":metrics["score"],"promoted":result["promoted"],"fold_deltas":result["fold_deltas"],"checks":checks,"bootstrap":bootstrap}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,required=True)
    run(parser.parse_args().plan.resolve())
