"""Archive terminal evidence and summarize a frozen relation-review experiment."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patientphex.data import digest_file, write_json


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    entity_review = "entity_index" in plan.get("identity_fields", [])
    report = ROOT / plan["report_dir"]
    queue = json.loads((report / "queue.json").read_text())
    if queue["status"] not in {"completed", "failed"}:
        raise ValueError("Wait for a terminal queue before publishing this report.")
    if queue["plan_sha256"] != digest_file(plan_path):
        raise ValueError("Queue does not belong to this frozen plan.")
    write_json(report / "plan.json", plan)
    task_count = 0
    for fold in plan["pilot_folds"]:
        directory = ROOT / plan["work_dir"] / f"fold{fold}"
        for name in ["prepared.json", "preflight.json", "summary.json", "progress.json"]:
            if (directory / name).exists():
                shutil.copyfile(directory / name, report / f"fold{fold}_{name}")
        task_count += json.loads((directory / "prepared.json").read_text())["task_count"]
    title = "实体概念含义核对实验结果" if entity_review else "患者归属复核实验结果"
    lines = ["# " + title, "", f"计划：`{plan_path.relative_to(ROOT).as_posix()}`，SHA256 `{digest_file(plan_path)}`。", "",
             "当前已确认的官方 B 分数仍为纯 CPU 文件的 0.6152；本报告不包含新官方成绩。", ""]
    if (report / "futility_stop.json").exists():
        stop = json.loads((report / "futility_stop.json").read_text())
        lines.extend(["因格式门槛已不可能达到而提前停止，未进行准确率评分，未生成 B 文件。即使剩余回答全部有效，下列一折也不能达到冻结的 98% 成功率。", "",
                      "| 折 | 已完成 / 计划 | 格式失败 | 最终成功率的理论上限 |", "|---|---:|---:|---:|"])
        for progress in stop["progress"]:
            lines.append(f"| {progress['fold']} | {progress['completed']} / {progress['total']} | {progress['format_errors']} | {progress['maximum_possible_format_success_rate']:.6f} |")
        lines.extend(["", "只终止了核对 PID、父进程、命令、工作目录和用户身份后的本轮 worker；已完成回答缓存和全部日志保留。失败证据在 `futility_stop.json`，队列退出状态在 `queue.json`。"])
    elif (report / "summary.json").exists():
        result = json.loads((report / "summary.json").read_text())
        if result["plan_sha256"] != digest_file(plan_path):
            raise ValueError("Evaluation summary belongs to a different plan.")
        status = f"两折通过，选择 `{result['selected']}`，仍需其余三折确认" if result["selected"] else "两折未通过晋级门槛，不扩展到 B"
        count_label = "实体评价单元" if entity_review else "关联"
        lines.extend([status + "。", "", f"| 方案 | 两折汇总分 | 两折平均增益 | 删除假{count_label} | 误删真{count_label} | 晋级 |", "|---|---:|---:|---:|---:|---|"])
        lines.append(f"| 固定上下文基线 | {result['baseline']['score']:.6f} | — | — | — | — |")
        for ranked in result["ranked"]:
            false_key, true_key = ("removed_false_mention_units", "removed_true_mention_units") if entity_review else ("removed_false_relations", "removed_true_relations")
            lines.append(f"| {ranked['policy']} | {ranked['metrics']['score']:.6f} | {ranked['mean_gain']:+.6f} | {ranked[false_key]} | {ranked[true_key]} | {'是' if ranked['eligible'] else '否'} |")
        change_description = ("仅删除原文词义与当前 HPO 概念明确不匹配的待核对实体，再由原冻结模型重算患者关联。NO、未映射、复合 ID、唯一精确 HPO 术语及不明确判断均保留；不改边界、不增实体、不重指派 HPO。" if entity_review else
                              "实体与 HPO 映射完全固定；仅删除有原文证据指向其他患者或一般描述的既有阳性关联。未映射表型与不明确判断保留。")
        lines.extend(["", change_description, "",
                      "| 折 | 任务数 | 格式成功率 | 生成耗时 / 秒 | 显存峰值 / GiB |", "|---|---:|---:|---:|---:|"])
        for evidence in result["inference"]:
            lines.append(f"| {evidence['fold']} | {evidence['task_count']} | {evidence['format_success_rate']:.6f} | {evidence['generation_seconds']:.1f} | {evidence['peak_allocated_gib']:.3f} |")
        lines.extend(["", f"最佳策略的描述性文献配对 bootstrap 95% 区间：`{result['best_bootstrap']['interval_95']}`。这是选择后的诊断，不能替代预设门槛。",
                      "", "逐折指标见 `fold*_metrics.json`，删除记录见 `*_edits.json`，严格输出校验见 `validation.json`。"])
    else:
        lines.extend(["执行失败，尚无完整评估结果。", "", f"错误：`{queue.get('error', 'unknown')}`。详情见队列与推理日志。"])
    lines.extend(["", "## 输入、门槛和边界", "",
                  f"使用固定 fold {plan['pilot_folds']}、{task_count} 条复核任务。模型 Qwen3-8B 在 177 服务器本地 FP16 推理，GPU 0 / 2，未微调，未调用外部推理 API。",
                  "", "提示词使用盲文档、原模型折外实体及比赛患者锚点，不含验证答案或示例；原关联模型标签来源检查及基线精确重放已通过。所有源文本、模型、计划及代码 SHA256 固定在计划中。",
                  "", ("两折门槛：平均增益至少 0.008，最差折损失不超过 0.002，四项 F1 损失不超过 0.002，实体 mention F1 必须提高，各折格式成功率至少 98%。通过后才固定策略确认其余三折；全五折门槛详见计划。" if entity_review else
                         "两折门槛：平均增益至少 0.008，最差折损失不超过 0.002，关联微/宏 F1 损失不超过 0.002，实体指标不变，各折格式成功率至少 98%。通过后才固定策略确认其余三折；全五折门槛详见计划。"),
                  "", "这些是反复使用的开发文献，既非独立测试，也不是官方 B 结果。不得据此声称已经达到 0.70。", "",
                  "## 复现和保存", "", f"原始模型回答和提示词指纹保存在 `{plan['work_dir']}/fold*/prompts/`，摘要已复制到本目录。", "",
                  "原环境与依赖验收说明：`reports/llm_extraction/remote_runbook.md`。所有服务器文件实际保存在 `/home/dcf/chip2026/`。", "",
                  "```bash", "cd /home/dcf/chip2026", f".venv/bin/python scripts/check_llm_association.py --plan {plan_path.relative_to(ROOT).as_posix()}", "```", "",
                  "已有工作目录和完成的评估受覆盖保护。未自动提交 Git、推送分支或向比赛平台提交。", ""])
    (report / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {path.relative_to(report).as_posix(): digest_file(path) for path in sorted(report.rglob("*"))
                if path.is_file() and path.name != "artifact_manifest.json"}
    write_json(report / "artifact_manifest.json", manifest)
    print(json.dumps({"report": str(report), "artifact_count": len(manifest), "queue_status": queue["status"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    run(parser.parse_args().plan.resolve())
