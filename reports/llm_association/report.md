# Qwen3 患者归属复核试验

状态：因原文证据校验失败次数使格式门槛不再可能达到，已提前停止，没有计算本轮准确率。当前 B 候选仍为 `submissions/patientphex_b_context.jsonl`，其官方 B 成绩未知。已确认的官方 0.6152 属于纯 CPU 文件。

## 提前终止及原因

fold 0 已完成 98/403 条、3 条证据失败；fold 1 已完成 103/297 条、6 条证据失败。后者即使剩余答案全部有效，最终格式成功率也最多为 97.9798%，低于预设的 98%。两种策略都无法晋级，因此在尚未查看准确率前停止，保留全部已完成回答及日志。

失败来自模型生成了不在所给片段中的附加引文，不能将这些引文当作原文。监控脚本验证工作进程的 PID、父进程、命令、目录和用户身份后，只向本轮两个 Python worker 发送 SIGTERM；两个 wrapper 均以 143 退出，队列记录终止，GPU 0、2 已释放。证据见 `futility_stop.json`。未修改格式门槛或解析器来挽救这一轮。

下一项试验将使用原文片段编号代替自由生成引文，保留精确来源可追溯性，重新冻结提示词与计划。此处的失败说明证据输出协议不够可靠，尚不能证明患者归属判断本身没有收益。

## 检验的问题

固定上下文关联方案的实体及 HPO 映射，用服务器本地 Qwen3-8B 复核已预测为阳性的患者—表型关联，区分目标患者表现、其他患者表现和疾病一般描述。本轮不增加关联，不修改实体，不复核未映射文本。

原文中没有明确归属、证据引用不精确或输出格式错误时保留原关联。模型不能凭疾病、基因或常识推断患者表型。提示词不包含训练示例、验证答案或基线概率。

## 冻结方案与预检

- 计划：`experiments/llm_association/plan.json`，SHA256 `e8fb7094ac71a33d0820b01602d23d66a332389a1b12aad0049394443e29346f`。
- 使用原固定五折中的 0、1 两折，共 32 篇文献；关联模型分别只使用其余 64 篇拟合，已检查标签来源隔离并精确重放基线预测。
- 700 条复核任务：fold 0 为 403 条，fold 1 为 297 条。
- 验证了全部 80 篇训练文献的 677 个患者锚点与原文一致；132 项回归测试通过。
- 真实 tokenizer 检查通过；两折最长输入分别为 1,747 / 1,884 tokens，均无需删减构造的原文片段。543 条任务纳入该表型的全部预测出现位置，其他任务只取距目标患者最近的三处。
- 同一已验收的 Qwen3-8B 参数及环境，FP16、eager、禁用 thinking、greedy、生成上限 256 tokens。未进行 Qwen 微调。环境说明与验收证据沿用 `reports/llm_extraction/remote_runbook.md`。
- GPU 2 执行 fold 0，GPU 0 执行 fold 1；各自检查其他计算进程并持有项目锁。所有文件实际位于 `/home/dcf/chip2026/`。

## 两个预设策略

1. `complete_only`：模型判为 `other_or_general`，且纳入全部预测出现位置、未因 token 预算删减片段时，才删除原关联。“全部出现位置”不代表阅读全文。
2. `all_reviewed`：只要模型判为 `other_or_general` 且引用精确匹配提供的原文，就删除原关联。

扩展门槛在评分前冻结：两折平均总分增益至少 0.008，最差折损失不超过 0.002，关联微/宏 F1 损失不超过 0.002，实体指标完全不变，每折格式成功率至少 98%。选择符合门槛且平均增益最高的策略；无策略通过时不扩展到 B。

通过两折后仍须固定策略，在其余三折确认：后三折平均增益至少 0.003，全五折总分增益至少 0.005，至少四折不下降，分量损失不超过 0.002，文献级配对 bootstrap 下界为正。反复使用的开发交叉验证不等于独立测试，也不能推算官方 B 分数。

## 执行和恢复

首次运行曾通过预检并启动 `chip-qwen-association` tmux 会话，现已终止。队列原定在两折结束后调用 CPU 评分，但格式门槛提前失败，未进入评分。`queue.json` 与 `futility_stop.json` 一起记录最终状态，不会自动晋级 B 或向平台提交。

```bash
cd /home/dcf/chip2026
.venv/bin/python scripts/check_llm_association.py
```

模型回答按提示词指纹逐条原子缓存于 `work/span_ner/llm_association/fold*/prompts/`。以下命令仅记录历史调用方法；本轮已确定不能晋级，不应自动续跑以浪费算力：

```bash
LLM_GPU_INDEX=2 bash scripts/run_llm_association.sh --fold 0
LLM_GPU_INDEX=0 bash scripts/run_llm_association.sh --fold 1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/run_llm_association.py evaluate
```

不要在旧任务仍运行时重复启动队列，不要覆盖已完成的评分。评分完成后执行 `scripts/audit_llm_association.py`，分开统计删除假关联和误删真关联，并与评分器计数核对。错误分析只用于后续假设，不能回改本轮门槛。
