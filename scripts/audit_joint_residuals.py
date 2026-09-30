"""Attribute remaining development errors without fitting or producing predictions."""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.association import _concepts
from patientphex.data import digest_file, read_jsonl, write_json
from patientphex.evaluation import _association_set, evaluate


def associations(document):
    return {row["patient_id"]: _association_set(row["phenotype"]) for row in document["association"]}


def run():
    source = ROOT / "PatientPheX-V1-A/PatientPheX-train.jsonl"
    before_path = ROOT / "reports/association_context/best_development_oof.jsonl"
    after_path = ROOT / "reports/joint_confirmation/development_oof.jsonl"
    split_path = ROOT / "reports/cpu_baseline/split.json"
    documents = read_jsonl(source)
    before = {str(d["pmc_id"]): d for d in read_jsonl(before_path)}
    after = {str(d["pmc_id"]): d for d in read_jsonl(after_path)}
    folds = json.loads(split_path.read_text())["folds"]
    fold_by_doc = {identifier: fold for fold, ids in enumerate(folds) for identifier in ids}
    if set(before) != set(after) or set(after) != set(fold_by_doc):
        raise ValueError("Development prediction and split coverage differ.")
    categories, transitions, unavailable_sources = Counter(), Counter(), Counter()
    by_group, by_document = {}, []
    for document in documents:
        identifier = str(document["pmc_id"])
        old, new = before[identifier], after[identifier]
        actual, prior, current = associations(document), associations(old), associations(new)
        available = {c for e in new["entities"] for c in _concepts(e)}
        previous_available = {c for e in old["entities"] for c in _concepts(e)}
        gold_any_patient = set().union(*actual.values())
        current_any_patient = set().union(*current.values())
        gold_entity_concepts = {c for e in document["entities"] for c in _concepts(e)}
        negated_candidates = {c for e in new["entities"] if e.get("note") == "NO"
                              for c in _concepts({**e, "note": ""})}
        local_counts = Counter()
        for patient, gold in actual.items():
            old_values, values = prior[patient], current[patient]
            for concept in sorted(gold - values):
                if concept not in available:
                    category = "fn:concept_removed_by_review" if concept in previous_available else "fn:concept_already_missing"
                    if concept in negated_candidates:
                        unavailable_sources["predicted_negated_only"] += 1
                    elif concept in gold_entity_concepts:
                        unavailable_sources["gold_positive_entity_not_available"] += 1
                    else:
                        unavailable_sources["association_concept_absent_from_gold_positive_entities"] += 1
                elif concept in current_any_patient:
                    category = "fn:available_assigned_to_other_patient"
                else:
                    category = "fn:available_assigned_to_no_patient"
                categories[category] += 1
                local_counts[category] += 1
            for concept in sorted(values - gold):
                if concept in gold_any_patient:
                    category = "fp:belongs_to_other_gold_patient"
                elif concept in gold_entity_concepts:
                    category = "fp:gold_entity_without_patient_association"
                else:
                    category = "fp:concept_absent_from_gold_positive_entities_and_associations"
                categories[category] += 1
                local_counts[category] += 1
            transitions["true_added"] += len((values - old_values) & gold)
            transitions["false_added"] += len((values - old_values) - gold)
            transitions["true_removed"] += len((old_values - values) & gold)
            transitions["false_removed"] += len((old_values - values) - gold)
        n = len(document["patient"])
        group = "1_patient" if n == 1 else "2_to_4_patients" if n <= 4 else "5_or_more_patients"
        by_group.setdefault(group, []).append(document)
        old_metrics, metrics = evaluate([document], [old]), evaluate([document], [new])
        by_document.append({"pmc_id": identifier, "fold": fold_by_doc[identifier], "patients": n,
                            "baseline_score": old_metrics["score"], "score": metrics["score"],
                            "score_delta": metrics["score"] - old_metrics["score"],
                            "association_macro_f1": metrics["association_macro"]["f1"],
                            "remaining_association_errors": dict(local_counts)})
    metrics = evaluate(documents, after.values())
    for kind in ["fp", "fn"]:
        assert sum(v for k, v in categories.items() if k.startswith(kind + ":")) == metrics["association_micro"][kind]
    groups = {}
    for group, docs in by_group.items():
        groups[group] = {"baseline": evaluate(docs, [before[str(d["pmc_id"])] for d in docs]),
                         "joint": evaluate(docs, [after[str(d["pmc_id"])] for d in docs])}
    result = {"scope": "Label-using diagnostics on repeatedly used development documents only. No B labels, fitted rule, selected threshold, oracle prediction export, or new performance claim.",
              "input_sha256": {p.relative_to(ROOT).as_posix(): digest_file(p) for p in [source, before_path, after_path, split_path]},
              "code_sha256": digest_file(Path(__file__)), "metrics": metrics,
              "association_error_categories": dict(categories), "association_transitions": dict(transitions),
              "unavailable_concept_sources": dict(unavailable_sources), "patient_count_groups": groups,
              "documents": sorted(by_document, key=lambda row: row["score_delta"])}
    write_json(ROOT / "reports/entity_residuals/joint_association_audit.json", result)
    missing = categories["fn:concept_removed_by_review"] + categories["fn:concept_already_missing"]
    reachable = categories["fn:available_assigned_to_other_patient"] + categories["fn:available_assigned_to_no_patient"]
    lines = ["# 联合方案剩余错误诊断", "",
             "对象：已完成全五折确认的联合方案；这些训练文献被开发反复使用。本报告使用真值归因错误，不训练模型、不导出使用真值修正的预测，也不读取 B 标签。", "",
             f"当前开发分数 {metrics['score']:.6f}。关联漏检 {metrics['association_micro']['fn']} 条，其中实体概念不可用 {missing} 条（{missing/metrics['association_micro']['fn']:.1%}），概念可用但未正确关联 {reachable} 条。", "",
             "| 关联错误来源 | 数量 |", "|---|---:|",
             f"| 原实体候选已缺失 | {categories['fn:concept_already_missing']} |",
             f"| 概念核对删除后不可用 | {categories['fn:concept_removed_by_review']} |",
             f"| 已有概念只分配给其他患者 | {categories['fn:available_assigned_to_other_patient']} |",
             f"| 已有概念未分配给任何患者 | {categories['fn:available_assigned_to_no_patient']} |",
             f"| 误关联到其他真值患者的概念 | {categories['fp:belongs_to_other_gold_patient']} |",
             f"| 真值实体存在，但没有对应患者关联 | {categories['fp:gold_entity_without_patient_association']} |",
             f"| 真值正实体和患者关联均没有该概念 | {categories['fp:concept_absent_from_gold_positive_entities_and_associations']} |", "",
             "最后一类不能仅凭标注缺失断言医学含义错误；可能包含错映射、泛化程度差异和标注口径问题。", "",
             "| 文献患者数 | 文献数 | 患者数 | 联合关联宏 F1 | 相比上下文基线 |", "|---|---:|---:|---:|---:|"]
    for group, values in groups.items():
        counts = values["joint"]["counts"]
        value = values["joint"]["association_macro"]["f1"]
        delta = value - values["baseline"]["association_macro"]["f1"]
        lines.append(f"| {group} | {counts['documents']} | {counts['patients']} | {value:.6f} | {delta:+.6f} |")
    lines.extend(["", "后续优先级：", "",
                  "1. 优先检查实体候选生成的召回缺口。先区分未见表述、边界错误和否定范围，再固定有针对性的两折试验；只有通过原完整验证流程才推广。现有结果不支持只靠关联阈值弥补大部分漏检。",
                  "2. 多患者文献继续检查患者提及与跨段归属。已有融合在该组收益较大，但其关联宏 F1 仍低于少患者文献；不直接按患者数修改预测阈值。",
                  "3. 先完成当前 B 候选与原始文件校验，并取得对应文件的官方反馈。上述归因只用于研究方向，不将开发真值中的个案规则写入 B 推理。", "",
                  "复现：`python scripts/audit_joint_residuals.py`；实体误差类型另见 `joint_audit.json`，由 `audit_entity_residuals.py --predictions reports/joint_confirmation/development_oof.jsonl --output reports/entity_residuals/joint_audit.json` 生成。", "",
                  "关联 FN/FP 的分类总数均与项目评分器逐项一致；数据和预测 SHA256 见 `joint_association_audit.json`。", ""])
    (ROOT / "reports/entity_residuals/joint_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["association_error_categories", "association_transitions", "unavailable_concept_sources"]}))
    print(json.dumps({group: {"documents": row["joint"]["counts"]["documents"], "score": row["joint"]["score"],
                             "association_macro_f1": row["joint"]["association_macro"]["f1"],
                             "association_macro_gain": row["joint"]["association_macro"]["f1"]-row["baseline"]["association_macro"]["f1"]}
                      for group, row in groups.items()}))


if __name__ == "__main__":
    run()
