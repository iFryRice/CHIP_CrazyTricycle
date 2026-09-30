"""Compare validated predictions on the same unlabeled B cohort, without scoring B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patientphex.association import _gold_concepts
from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.ontology import Ontology
from patientphex.validation import validate_submission


def units(document: dict) -> tuple[set, set]:
    mentions = {(entity["offset"], entity["length"], identifier)
                for entity in document["entities"] if str(entity.get("note") or "").upper() != "NO"
                for identifier in entity["identifier"].split(";")}
    links = {(row["patient_id"], concept) for row in document["association"]
             for concept in _gold_concepts(row["phenotype"])}
    return mentions, links


def agreement(left: set, right: set) -> dict:
    union = left | right
    return {"cpu_units": len(left), "candidate_units": len(right), "shared": len(left & right),
            "cpu_only": len(left - right), "candidate_only": len(right - left),
            "jaccard_agreement": len(left & right) / len(union) if union else 1.0}


def run(args: argparse.Namespace) -> dict:
    output = args.report_dir
    if output.exists():
        raise FileExistsError("B comparison report already exists")
    target_path = ROOT / "PatientPheX-V1-B/PatientPheX-V1-B.jsonl"
    targets = read_jsonl(target_path)
    if len(targets) != 100 or sum(len(d["patient"]) for d in targets) != 244:
        raise ValueError("Wrong B cohort")
    if any(d.get("entities") or d.get("association") for d in targets):
        raise ValueError("This comparison only accepts an unlabeled B target")
    target_hash = digest_file(target_path)
    ontology = Ontology(ROOT / "PatientPheX-V1-A/hp.obo")
    cpu_validation = validate_submission(args.cpu, targets, ontology)
    candidate_validation = validate_submission(args.candidate, targets, ontology)
    cpu_metrics = json.loads((ROOT / "reports/cpu_b_control/metrics.json").read_text(encoding="utf-8"))
    candidate_metrics = json.loads((ROOT / "reports/association_reweighting/comparison.json").read_text(encoding="utf-8"))
    candidate_plan = json.loads((ROOT / "experiments/association_reweighting/plan.json").read_text(encoding="utf-8"))
    if cpu_metrics["target_set"] != "B" or cpu_metrics["target_has_gold"] or cpu_metrics["target_score"] is not None:
        raise ValueError("CPU result must be a fresh B prediction with no claimed B score")
    if "Fresh four-configuration" not in cpu_metrics["run_mode"]:
        raise ValueError("Require the newly rerun CPU control")
    if cpu_metrics["input_sha256"][cpu_metrics["target_path"]] != target_hash or candidate_plan["b_target"]["sha256"] != target_hash:
        raise ValueError("Methods did not use the same B input")
    if cpu_metrics["input_sha256"][cpu_metrics["training_path"]] != candidate_plan["training"]["sha256"]:
        raise ValueError("Methods did not use the same labeled training input")
    if cpu_validation["sha256"] != cpu_metrics["submission_sha256"]:
        raise ValueError("CPU submission differs from its fresh run")
    if candidate_validation["sha256"] != candidate_metrics["b_result"]["submission"]["sha256"]:
        raise ValueError("Candidate submission differs from its experiment")
    cpu, candidate = read_jsonl(args.cpu), read_jsonl(args.candidate)
    by_cpu, by_candidate = ({str(d["pmc_id"]): d for d in records} for records in (cpu, candidate))
    pooled_cpu = (set(), set())
    pooled_candidate = (set(), set())
    per_document = []
    for doc in targets:
        identifier = str(doc["pmc_id"])
        left, right = units(by_cpu[identifier]), units(by_candidate[identifier])
        per_document.append({"pmc_id": identifier, "positive_mentions": agreement(left[0], right[0]),
                             "patient_relations": agreement(left[1], right[1])})
        for index in (0, 1):
            pooled_cpu[index].update((identifier, *unit) for unit in left[index])
            pooled_candidate[index].update((identifier, *unit) for unit in right[index])
    result = {"status": "completed", "scope": "Same B input and training data; output agreement only, not B accuracy.",
              "target_sha256": target_hash, "official_B_scores_available": False, "B_accuracy_computed": False,
              "B_parameter_selection_performed": False,
              "cpu": {"path": str(args.cpu), "validation": cpu_validation,
                      "selected_configuration": cpu_metrics["selected_configuration"],
                      "fresh_training": True, "development_metrics": cpu_metrics["selected_metrics"]},
              "candidate": {"path": str(args.candidate), "validation": candidate_validation,
                            "configuration": "BiomedBERT/HPO/UMLS frozen entities + patient-balanced association negative weight 0.5",
                            "development_metrics": candidate_metrics["candidate"]},
              "positive_mention_agreement": agreement(pooled_cpu[0], pooled_candidate[0]),
              "patient_relation_agreement": agreement(pooled_cpu[1], pooled_candidate[1]),
              "per_document": per_document,
              "limitations": ["More predictions or greater agreement does not imply higher accuracy.",
                              "CPU and candidate are complete pipelines with different entity resources and association models; this is not a single-component ablation.",
                              "Development scores use the same reused training documents and are not B leaderboard scores.",
                              "B leaderboard results remain unknown until official feedback for these exact file hashes."]}
    output.mkdir(parents=True)
    write_json(output / "comparison.json", result)
    lines = ["# 同一 B 集上的 CPU 对照与关联优化候选", "",
             "CPU 已重新完成四配置五折验证及全部 80 篇训练文献的拟合，再预测 B 集；没有复用旧 CPU 折外预测或 A 榜输出。两套方法使用相同的 B 输入和相同的监督训练集。", "",
             "**尚无 B 榜官方分数。以下 B 统计和一致性不能说明哪套方案更准确。**", "",
             "| B 输出统计 | 重跑 CPU | 关联优化候选 |", "|---|---:|---:|"]
    for key, label in (("documents", "文献"), ("patients", "患者"), ("entities", "实体"),
                       ("association_pairs", "患者关联"), ("negated_entities", "否定实体"), ("unmapped_entities", "未映射实体")):
        lines.append(f"| {label} | {cpu_validation[key]} | {candidate_validation[key]} |")
    lines += ["", f"CPU 方案：`{cpu_metrics['selected_configuration']}`。两份文件均通过严格格式、原文跨度、患者引用和完整覆盖检查。", "",
              "CPU 文件：`submissions/patientphex_b_cpu.jsonl`；关联优化候选：`submissions/patientphex_b_association_reweighted.jsonl`。", "",
              f"CPU SHA-256：`{cpu_validation['sha256']}`。", "",
              f"关联优化候选 SHA-256：`{candidate_validation['sha256']}`。", "",
              "## 重新计算的开发成绩（不是 B 成绩）", "",
              "| 方法 | 80 篇训练文献上的五折开发近似总分 |", "|---|---:|",
              f"| 本次重新运行的 CPU | {cpu_metrics['selected_metrics']['score']:.6f} |",
              f"| 本轮关联优化 | {candidate_metrics['candidate']['score']:.6f} |", "",
              "这些分数用于开发与选型；由于 B 没有本地标签，不能将它们写成 B 榜成绩。若 CPU 开发分数与旧记录一致，原因是监督数据、划分与算法保持相同，而非复用了 A 集预测。", "",
              "## B 预测一致性", ""]
    for key, label in (("positive_mention_agreement", "正向实体单元"), ("patient_relation_agreement", "患者—概念关联")):
        item = result[key]
        lines.append(f"{label}：共同 {item['shared']}，仅 CPU {item['cpu_only']}，仅关联优化 {item['candidate_only']}；Jaccard 一致性 {item['jaccard_agreement']:.4f}。")
        lines.append("")
    lines += ["这些是输出差异，不是正确/错误计数。没有据此调整任何模型参数或进行融合。", "",
              "CPU 完整训练记录见 `reports/cpu_b_control/`，关联优化记录见 `reports/association_reweighting/`。实际 B 榜优劣需要官方针对上述具体文件的评分。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"status": "completed", "B_accuracy_computed": False,
                      "cpu_validation": cpu_validation, "candidate_validation": candidate_validation,
                      "patient_relation_agreement": result["patient_relation_agreement"]}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, default=ROOT / "submissions/patientphex_b_cpu.jsonl")
    parser.add_argument("--candidate", type=Path, default=ROOT / "submissions/patientphex_b_association_reweighted.jsonl")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/b_method_comparison")
    run(parser.parse_args())
