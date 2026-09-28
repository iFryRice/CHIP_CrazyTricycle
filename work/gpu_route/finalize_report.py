"""Verify the completed fusion and export comparison documentation only."""

import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fuse_predictions import fuse, complete_predictions
from patientphex.data import read_jsonl, digest_file
from patientphex.ontology import Ontology
from patientphex.validation import validate_submission


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


out = Path(r"D:\Cache\Codex\2026-09-27\cha-xu\outputs")
report_dir = ROOT / "reports/gpu_comparison"
search = load(ROOT / "reports/fusion_search/comparison.json")
best = search["candidates"][0]
assert best["method"] == "cpu_union_gpu" and best["multi_patient_only"]
cpu_path = ROOT / "work/gpu_route/cpu_a_input.jsonl"
cpu = read_jsonl(cpu_path)
source = read_jsonl(ROOT / "PatientPheX-V1-A/PatientPheX-A.jsonl")
summary_path = ROOT / "reports/gpu_a/summary.json"
gpu = complete_predictions(Path(load(summary_path)["output"]), summary_path,
                           cpu_path, {d["pmc_id"] for d in cpu})
fusion_path = ROOT / "submissions/patientphex_a_cpu_gpu_union_multi.jsonl"
fusion = read_jsonl(fusion_path)
assert fusion == fuse(cpu, source, "cpu_union_gpu", gpu=gpu, multi_patient_only=True)
expected_hash = "66a1ddb7910dd2f06d35470cd1b65e2df65054819bad730350b5dd0ef32f3b50"
assert digest_file(fusion_path) == expected_hash
validation = validate_submission(fusion_path, source, Ontology(ROOT / "PatientPheX-V1-A/hp.obo"))
assert digest_file(out / "patientphex_a_fusion.jsonl") == expected_hash
plain = load(ROOT / "reports/gpu_validation/diagnostic_summary.json")
rag = load(ROOT / "reports/gpu_rag_validation/diagnostic_summary.json")
now = datetime.now().astimezone().isoformat(timespec="seconds")
lines = [
    "# PatientPheX 最终比较与交付报告", "", f"生成时间：{now}。", "",
    "推荐候选采用统一规则：**单患者文献完整保留CPU；多患者文献保留CPU关联，并补充普通GPU关联（CPU ∪ GPU）**。最终融合不使用RAG输出。",
    "全80篇本地诊断总分由0.639727提高至0.640983；排除原8篇pilot的72篇由0.633688提高至0.635222。提升较小。**这是12规则选择后的本地诊断，不是独立测试，不保证官方A榜改善。**",
    "本次没有使用UMLS，没有新增训练CPU或大模型权重。", "",
    "## 交付与复验", "",
    "推荐提交：`patientphex_a_fusion.jsonl`；同时保留CPU、普通GPU和RAG结果。",
    f"已从冻结CPU和完整普通GPU输出独立重算融合，逐条等于提交结果。严格校验通过：20篇、53患者、1335实体、514关联，{validation['bytes']:,}字节，warnings为空。",
    "实体、PMID、文献顺序不变；全部单患者记录保留CPU；多患者执行同一集合规则。没有按目标gold逐篇选择答案。",
    f"SHA256：`{expected_hash}`。",
    "A集没有gold，本次仅验证结构、覆盖、跨度、HPO和融合规则；未计算A集准确率。", "",
    "## 四项F1与总分", "",
]
metric_keys = ("mention", "document", "association_micro", "association_macro")
for scope, label in (("all_80", "全部80篇折外预测"), ("non_pilot_72", "非pilot72篇")):
    lines += [f"### {label}", "", "| 路线 | 实体F1 | 文档F1 | 关联Micro F1 | 关联Macro F1 | 总分 |",
              "|---|---:|---:|---:|---:|---:|"]
    rows = [("CPU", best["scopes"][scope]["all"]["baseline"]),
            ("普通GPU", plain["scopes"][scope]["candidate"]),
            ("RAG", rag["scopes"][scope]["candidate"]),
            ("最终融合", best["scopes"][scope]["all"]["candidate"])]
    for name, scores in rows:
        lines.append("| " + name + " | " + " | ".join(f"{scores[k]['f1']:.6f}" for k in metric_keys)
                     + f" | {scores['score']:.6f} |")
    lines.append("")
lines += ["72篇仍属于反复用于诊断和选型的训练资料，不能视为新独立测试集。实体被冻结，所以实体和文档F1保持不变。", "",
          "## 12种确定性融合候选", "",
          "| 规则 | 仅多患者融合 | 全80总分 | 相对CPU | 非pilot72总分 | 相对CPU |",
          "|---|---|---:|---:|---:|---:|"]
names = {"cpu_union_gpu": "CPU∪GPU", "cpu_intersection_gpu": "CPU∩GPU",
         "cpu_union_rag": "CPU∪RAG", "cpu_intersection_rag": "CPU∩RAG",
         "majority": "三路多数票", "cpu_plus_consensus": "CPU∪(GPU∩RAG)"}
for row in search["candidates"]:
    a, b = row["scopes"]["all_80"]["all"], row["scopes"]["non_pilot_72"]["all"]
    lines.append(f"| {names[row['method']]} | {'是' if row['multi_patient_only'] else '否'} | "
                 f"{a['candidate']['score']:.6f} | {a['score_delta']:+.6f} | "
                 f"{b['candidate']['score']:.6f} | {b['score_delta']:+.6f} |")
lines += ["", "选择依据为全80总分；72篇用于一致性检查。最佳规则在两个口径中均排名第一。", "",
          "## 分组与取舍", "",
          "| 范围 | 患者组 | CPU总分 | 融合总分 | 差值 | 改善/恶化/同分篇数 |",
          "|---|---|---:|---:|---:|---|"]
for scope, label in (("all_80", "全80"), ("non_pilot_72", "非pilot72")):
    for group, name in (("all", "全部"), ("single", "单患者"), ("multi", "多患者")):
        row = best["scopes"][scope][group]
        counts = row["document_outcomes"]
        lines.append(f"| {label} | {name} | {row['baseline']['score']:.6f} | {row['candidate']['score']:.6f} | "
                     f"{row['score_delta']:+.6f} | {counts['improved']}/{counts['worsened']}/{counts['unchanged']} |")
lines += ["", "全80中融合增加31条正确关联，也增加51条错误关联。召回提高而精确率下降，Micro和Macro F1均小幅改善，但并非每篇文章改善。", "",
          "## 实测运行耗时", "", "| 阶段 | 最后调用耗时 | 新请求/缓存说明 |", "|---|---:|---|"]
timings = []
for name, relative in (
    ("CPU完整5折", "work/gpu_route/cpu_metrics_input.json"),
    ("普通GPU pilot", "reports/gpu_validation_pilot/summary.json"),
    ("普通GPU全80续跑", "reports/gpu_validation/summary.json"),
    ("普通GPU A集最终修复", "reports/gpu_a/summary.json"),
    ("RAG pilot", "reports/gpu_rag_pilot/summary.json"),
    ("RAG全80续跑", "reports/gpu_rag_validation/summary.json"),
    ("RAG A集", "reports/gpu_rag_a/summary.json"),
):
    summary = load(ROOT / relative)
    seconds = summary.get("runtime_seconds", summary.get("elapsed_seconds"))
    new = summary.get("new_requests")
    desc = "完整CPU流水线" if new is None else f"{new}个新请求，{summary['documents'] - new}篇已完成缓存"
    lines.append(f"| {name} | {seconds:.3f}秒 | {desc} |")
    timings.append({"stage": name, "seconds": seconds, "new_requests": new, "source": relative})
lines += ["", "普通GPU/RAG全80续跑各复用8篇pilot。普通GPU A集最后summary只记录1篇严格schema修复，12.055秒不是20篇全程时间。阶段可能并行、共享服务，耗时不能直接相加解释为整个任务墙钟时间。最终融合只做本地集合运算，没有新增推理。", "",
          "## 方法与复现说明", "",
          "以下纳入本次methods.md的方法说明，并更新最终完成状态。完整12候选和分组四项指标见项目reports/fusion_search/comparison.json。", ""]
completion = {}
for name, relative in (("plain80", "reports/gpu_validation/summary.json"),
                       ("plainA", "reports/gpu_a/summary.json"),
                       ("rag80", "reports/gpu_rag_validation/summary.json"),
                       ("ragA", "reports/gpu_rag_a/summary.json")):
    path = ROOT / relative
    summary = load(path)
    assert summary["all_documents_refined"] and summary["cpu_documents"] == 0
    assert len(summary["status"]) == summary["documents"] and all(s["source"] == "gpu" for s in summary["status"])
    audits = []
    for file in path.parent.glob("*.json"):
        record = load(file)
        if isinstance(record, dict) and record.get("pmc_id") is not None and record.get("source") == "gpu":
            audits.append(record)
    assert len({a["pmc_id"] for a in audits}) == summary["documents"]
    assert all(a.get("prompt_report", {}).get("source_is_excerpted") is False for a in audits)
    completion[name] = {"documents": summary["documents"], "all_documents_refined": True,
                        "cpu_fallback_documents": 0, "excerpted_documents": 0}
methods = (report_dir / "methods.md").read_text(encoding="utf-8")
methods = methods.replace("开关作为待实现并比较的候选策略；是否纳入最终方案，以最终 CLI 核验和统一比较结果为准。",
                          "开关已实现并参与12候选比较；最终采用仅多患者执行CPU与普通GPU并集。")
methods = methods.replace("RAG 完整性需以最终完成后的逐篇记录核验，不能仅由参数值推定。",
                          "RAG 80篇验证及20篇A集已逐篇核验完成，均未截断，最终全部使用GPU结果。")
lines.append(methods)
report = "\n".join(lines).rstrip() + "\n"
report_dir.mkdir(parents=True, exist_ok=True)
(report_dir / "final_report.md").write_text(report, encoding="utf-8")
(out / "patientphex_comparison_report.md").write_text(report, encoding="utf-8")
artifacts = []
for name in ("patientphex_a_cpu.jsonl", "patientphex_a_gpu.jsonl", "patientphex_a_rag.jsonl",
             "patientphex_a_fusion.jsonl", "patientphex_comparison_report.md"):
    path = out / name
    artifacts.append({"name": name, "path": str(path), "bytes": path.stat().st_size, "sha256": digest_file(path)})
manifest = {
    "generated_at": now, "recommended_submission": "patientphex_a_fusion.jsonl",
    "method": "cpu_union_gpu", "multi_patient_only": True,
    "independent_rule_recomputation_equal": True, "entities_pmid_order_preserved": True,
    "UMLS_used": False, "is_independent_test": False, "official_A_score_known": False,
    "validation": validation, "artifacts": artifacts, "completion_audit": completion,
    "timings": timings, "project_report": str(report_dir / "final_report.md"),
    "fusion_diagnostics": str(ROOT / "reports/fusion_search/comparison.json"),
    "scores": {scope: {"cpu": best["scopes"][scope]["all"]["baseline"]["score"],
                        "fusion": best["scopes"][scope]["all"]["candidate"]["score"]}
               for scope in ("all_80", "non_pilot_72")},
}
payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
(report_dir / "final_manifest.json").write_text(payload, encoding="utf-8")
(out / "patientphex_manifest.json").write_text(payload, encoding="utf-8")
print(json.dumps({"completed": True, "artifacts": artifacts,
                  "manifest": str(out / "patientphex_manifest.json"),
                  "manifest_sha256": digest_file(out / "patientphex_manifest.json")},
                 ensure_ascii=False, indent=2))
