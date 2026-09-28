# B 榜训练与候选结果

2026-09-29 北京时间 00:06 已生成 B 候选，随后完成服务器独立验收和本机复验。文件为 `submissions/patientphex_b.jsonl`，尚未上传比赛平台。

本次完整流水线已跑通，但新方案五折开发近似总分 **0.601418** 低于冻结 CPU 对照 **0.639727**，差值 **−0.038309**。这份文件应视为可复现的实验候选，不能据此声称效果升级。B 集无本地答案，不能给出 B 榜成绩。

## 产物与校验

| 项目 | 结果 |
|---|---:|
| B 文献 / 患者 | 100 / 244 |
| 实体 / 患者关联 | 7,144 / 2,097 |
| 否定实体 / 未映射实体 | 173 / 27 |
| 文件大小 | 826,999 bytes |
| 文献顺序、患者覆盖、原文跨度、HPO 分支、格式 | 全部通过 |
| 校验警告 | 0 |

提交文件 SHA-256：`9256fbb6ac7aaabdd27bda3190d353ebd6b08da887b4065f86e75eafde7fc35c`。本机 `local_validation.json`、服务器 `submission_validation.json` 与独立重新执行的校验一致。

服务器提交路径：`/home/dcf/chip2026/submissions/patientphex_b.jsonl`。最终检查点：`/home/dcf/chip2026/work/span_ner/final-cap100-all80/last.pt`；其 SHA-256 为 `cf201805a7bcb1f4d066aa17d4bd1ebfbc076d6b81e66c1e2bedb92955b62e9f`。患者关联模型保存在 `/home/dcf/chip2026/work/span_ner/b_linking/patient_linking.pkl`。大模型和授权词库未复制进本机 Git 成果目录。

## 训练与资源

使用官方 80 篇训练文献及原有文献级五折划分。B 输入是 `PatientPheX-V1-B/PatientPheX-V1-B.jsonl`，只作推理；没有用 B 选阈值、训练或估算成绩。HPO 固定为 2026-06-23 的 `HP:0000118` 后代，排除根节点。UMLS/HPO 原始数据实际保存在 `/home/dcf/umls/`，训练和派生缓存实际保存在 `/home/dcf/chip2026/`。

BiomedBERT 模型版本为 `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` 的 `e1354b7a3a09615f6aba48dfad4b7a613eef7062`。跨度头保留重叠和否定，使用 FP16、长度 384、步幅 128、最大跨度 64、批量 4、梯度累积 4、正类权重上限 100。完整参数和源码/资源指纹见 `experiments/span_ner/selected_pipeline.json`。

首折比较原始权重、权重上限 100、边界排序损失后，按完整四指标选出权重上限 100 的方案。冻结跨度阈值 0.9702336192131042、`knowledge_union` 实体策略和患者关联阈值 0.3，随后在其余四折仅评价这一配置。确认折只在第 5 轮评价；其比较 JSON 中的 `checkpoint_validation_selection=true` 是评价器保守固定标记，实际训练清单均为 `eval_every_epoch=false`、`best_epoch=5`，不存在多轮挑选。首折仍用于选型，全部结果均属于开发估计。

最终模型对全部 80 篇文献固定训练 5 轮，共 910 步，耗时 553.13 秒，无验证集及按指标选点。实际检查点核验 `epoch=5`、`global_step=910`、`next_batch=0`，最终模型/阈值/代码/资源来源绑定全部通过。B 跨度推理耗时 17.25 秒，规范化与患者关联等最终处理耗时 18.87 秒。

训练峰值已分配显存约 3.076 GiB，B 推理约 0.762 GiB。完成后的服务器快照：251.61 GiB 总内存、234.87 GiB 可用内存、60.52 GiB 剩余磁盘；GPU 0/2 已回到空闲显示占用。本流程使用流式词库导出与轻量缓存，无需把整个 36 GB UMLS 装入内存。SQLite 读取继续使用服务器 `/usr/bin/python3`。

## 开发评估

采用同一近似本地 scorer，读取既有 CPU 折外预测重新计分，没有重跑 CPU 训练或推理。

| 指标 | 新方案 | 冻结 CPU | 差值 |
|---|---:|---:|---:|
| 实体提及 F1 | 0.636302 | 0.686909 | −0.050607 |
| 文献概念 F1 | 0.716036 | 0.733802 | −0.017766 |
| 患者关联 micro F1 | 0.558504 | 0.601903 | −0.043399 |
| 患者关联 macro F1 | 0.494830 | 0.536294 | −0.041465 |
| 四项平均 | 0.601418 | 0.639727 | −0.038309 |

五折分数依次为 0.642704、0.595890、0.590249、0.596919、0.578012；首折的小幅领先未在其余四折延续。单患者文献为 0.620608 对 0.666774，多患者文献为 0.598204 对 0.633075，两组均退化。排除选型首折后，剩余 64 篇汇总为 0.590159 对 0.640909，但仍不是独立测试。

没有官方 scorer；`-1`、正文范围等近似假设完整保存在 `development_summary.json`。同一 80 篇文献已反复用于开发，不能据此声明统计显著性，也不能与官方 A 榜成绩直接比较。不同组件同时变化，整体分差不能作为某个组件的因果效果。

## 问题与下一步

与 CPU 相比，实体提及仅增加 28 个 TP，却增加 927 个 FP；患者关联减少 97 个 TP，同时增加 60 个 FP。错误审计还发现，新候选已覆盖更多正确概念，但概念已在候选中、仍未关联到正确患者的漏检从 263 增至 392。因此下一轮优先固定已保存的候选，单独改进患者关联训练与决策，并用相同文献划分做单变量验证；目前未启动这轮新实验。

实体误报也需要后续处理，但不能把最终结果全归因于神经跨度阈值：首折组件审计中，约 91.6% 保留的正实体单位来自固定知识库，神经新增贡献为 46 TP / 50 FP。完整错误分类与证据见 `error_analysis.json`，首折限定审计见 `reports/span_ner/component_audit.json`。

## 复现与工程验收

- 冻结选择：`experiments/span_ner/selected_pipeline.json`，SHA-256 `15ca6b31e575429444f00f3014c6d20124835ea83b59f621ecc365cfc2154994`。
- 训练日志及输入指纹：`reports/span_ner/final-cap100-all80/`；B 推理清单：`reports/span_ner/b_inference_manifest.json`。
- 完整五折汇总：`development_summary.json` 与 `development_oof.jsonl`；B 配置与来源：`manifest.json`。
- 服务器环境执行 `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -q`，85 项全部通过，耗时 3.721 秒。
- 本机同套测试 74 项通过、11 项因未安装可选 torch 依赖而跳过；这些张量相关测试已在服务器环境实际通过。
- `scripts/complete_b_pipeline.sh` 记录本次训练和推理顺序；输出目录不可覆盖，不能直接重跑已有任务。校验现有 B 文件的只读命令见根目录 README。

所有原比赛输入和历史 A 榜结果保留。未提交比赛平台，未执行 Git commit 或 push。
