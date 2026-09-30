# 患者归属复核实验结果

计划：`experiments/llm_association_references/plan.json`，SHA256 `fa76180ac6dce03f800fb1b919491395dd233d28fe45623f0e7b0d6aa272d7c6`。

当前已确认的官方 B 分数仍为纯 CPU 文件的 0.6152；本报告不包含新官方成绩。

因格式门槛已不可能达到而提前停止，未进行准确率评分，未生成 B 文件。即使剩余回答全部有效，下列一折也不能达到冻结的 98% 成功率。

| 折 | 已完成 / 计划 | 格式失败 | 最终成功率的理论上限 |
|---|---:|---:|---:|
| 0 | 91 / 403 | 9 | 0.977667 |
| 1 | 93 / 297 | 1 | 0.996633 |

只终止了核对 PID、父进程、命令、工作目录和用户身份后的本轮 worker；已完成回答缓存和全部日志保留。失败证据在 `futility_stop.json`，队列退出状态在 `queue.json`。

## 输入、门槛和边界

使用固定 fold [0, 1]、700 条关联复核任务。模型 Qwen3-8B 在 177 服务器本地 FP16 推理，GPU 0 / 2，未微调，未调用外部推理 API。

提示词使用盲文档、原模型折外实体及比赛患者锚点，不含验证答案或示例；原关联模型标签来源检查及基线精确重放已通过。所有源文本、模型、计划及代码 SHA256 固定在计划中。

两折门槛：平均增益至少 0.008，最差折损失不超过 0.002，关联微/宏 F1 损失不超过 0.002，实体指标不变，各折格式成功率至少 98%。通过后才固定策略确认其余三折；全五折门槛详见计划。

这些是反复使用的开发文献，既非独立测试，也不是官方 B 结果。不得据此声称已经达到 0.70。

## 复现和保存

原始模型回答和提示词指纹保存在 `work/span_ner/llm_association_references/fold*/prompts/`，摘要已复制到本目录。

原环境与依赖验收说明：`reports/llm_extraction/remote_runbook.md`。所有服务器文件实际保存在 `/home/dcf/chip2026/`。

```bash
cd /home/dcf/chip2026
.venv/bin/python scripts/check_llm_association.py --plan experiments/llm_association_references/plan.json
```

已有工作目录和完成的评估受覆盖保护。未自动提交 Git、推送分支或向比赛平台提交。
