# 177 首轮实体训练结果

2026-09-28（Asia/Shanghai）。本阶段已完成服务器配置、资源核验、UMLS/HPO 接收检查和 BiomedBERT 首轮训练。训练流程可用，但当前模型误报过多，尚不能替换现有比赛提交。用户已明确当前是 **B 榜阶段**，后续推理和提交目标为 `PatientPheX-V1-B/PatientPheX-V1-B.jsonl`。

监督训练仍使用 `PatientPheX-V1-A/PatientPheX-train.jsonl` 中官方提供的 80 篇带标签文献；文件夹含 A 不代表模型面向 A 榜。B 集没有实体和关联答案，只作为最终推理输入，不能作为有标签训练数据或用于本地选阈值。本阶段未对 A 集生成新预测，且尚未生成 B 集提交。

B 集包含 100 篇唯一文献、244 名患者，与训练集和 A 集无文献交叉；原文件大小 2,027,053 字节，SHA-256 为 `4dafea5aa59bfd5f90151ee3732ee475323512b8691f370f4d4678b628cdb62a`。数据角色与后续 B 输出位置记录在 `experiments/span_ner/task.json`；这份清单不代表 HPO 映射、患者关联或最终推理已经执行。

B 原文件已同步至 `/home/dcf/chip2026/PatientPheX-V1-B/PatientPheX-V1-B.jsonl`，服务器上的字节哈希、文献数、患者数、空标签及与训练集的隔离检查全部通过，见 `b_target_validation.json`。旧 CPU `run/validate` 入口仍硬编码 A，新 span trainer 的 `--data` 是有标签监督输入，不能将 B 传给该参数。后续 B 推理和格式验证必须显式使用本清单中的 B 路径。

## 存储与资源

所有服务器文件均实际位于 `/home/dcf`，不是指向 `/disk2` 的符号链接：

| 内容 | 服务器路径 |
|---|---|
| 用户上传的 UMLS | `/home/dcf/umls/` |
| 比赛指定版 HPO | `/home/dcf/umls/hpo/2026-06-23/` |
| 隔离训练项目、环境、模型与日志 | `/home/dcf/chip2026/` |
| 首轮最佳模型 | `/home/dcf/chip2026/work/span_ner/fold0/best.pt` |
| 可续训状态 | `/home/dcf/chip2026/work/span_ner/fold0/last.pt` |

22:47 的服务器快照：物理内存约 251.6 GiB，可用约 216.3 GiB；`/home/dcf` 所在文件系统剩余约 72.0 GiB。完全空闲内存较低主要是文件缓存，不能据此判断内存不足。GPU 2 已结束本次作业，仅剩 14 MiB 图形进程占用。其他 GPU 有既有作业，未修改或停止。

UMLS 的 2,985 个文件、36,031,260,551 字节已独立核验存在性和大小；全量 SHA-256 通过来自上传方，本任务未重复扫描全部数据。6 个 HPO 文件的哈希及 HPO SQLite 完整性已独立验证。指定 HPO 本体与本项目文件一致。原 UMLS 索引是英语、十来源子集，不能当作全部 UMLS；最终输出以指定 HPO 版本与比赛允许概念为准。SQLite 使用 `/usr/bin/python3` 只读查询，无需把全部 RRF 载入内存。

## 实际训练与验证

- 模型：Microsoft BiomedBERT，固定 revision `e1354b7a3a09615f6aba48dfad4b7a613eef7062`，6 个官方文件校验通过。
- 环境：项目内 Python 3.12.13、torch 2.6.0+cu124、transformers 4.49.0；GPU 2，V100 32 GiB，FP16，2 个 CPU 线程。
- 任务：原文字符跨度与 positive/NO 标签；保留重叠实体。首轮没有使用 UMLS 特征，也没有训练 HPO 映射或患者关联模型。
- 数据：沿用原文献级 fold 0，64 篇训练、16 篇验证，无文献交叉；未使用 A/B 答案或外部人工标注表型数据。
- 配置：5 epochs，batch 4，梯度累积 4，窗口长度 384，stride 128，跨度宽度 64，seed 20260928。
- 覆盖：80 篇共 6,027 个原始实体，6,023 个可精确编码。4 个子词边界不对齐实体全部位于训练部分；相关负例已屏蔽，原始 gold 未改动。验证部分无此问题。
- 独立验收：48 项测试全部通过、无跳过；CUDA FP16 前向及反向计算通过；20 次有效更新的短训练通过。
- 正式结果：22:40:38 完成全部 5 epochs，耗时 490.209 秒；695 次更新尝试、692 次实际更新，差额是首轮 3 次 AMP 溢出保护跳步，后 4 轮无跳步。所有记录的 loss 有限；峰值分配显存 3.05750 GiB。

## 效果与问题

默认阈值 0.5、最佳 epoch 5 的字符跨度与否定标签 micro 指标为：

| Precision | Recall | F1 | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 5.886% | 93.284% | 11.073% | 1,139 | 18,212 | 82 |

这说明运行成功不等于效果可用。损失给正例的权重达到 1000，0.5 的模型分数不对应通常的等代价分类阈值，因此输出偏向召回。保存的预测包含每个标签的分数，可对不低于 0.5 的阈值作离线诊断，不必再运行 GPU 推理。

在同一验证集选择共同阈值 0.998051524 后，precision/recall/F1 为 21.530%/54.873%/30.925%。仍有 2,442 个误报，其中 2,158 个（88.4%）与同标签 gold 有字符重叠但边界不一致。阈值调整只能解决部分问题，下一步需要重点核查边界学习。不得据此删除真实重叠实体或更改 gold。完整诊断和使用限制见 `fold0/threshold_diagnostics.json`。

本折验证用于选择检查点和诊断阈值，不是独立测试。这里的分数不包含 HPO 归一化或患者关联，不能与比赛总分、官方实体指标或旧模型总分直接比较。本阶段未生成新的比赛提交，也未覆盖既有 CPU/GPU/RAG/融合成果。

后续应先检查高分错误跨度与边界判别，在训练文献范围内比较有依据的损失或困难负例方案，再接入指定 HPO/UMLS 术语映射与患者关联；最终使用 B 集推理并按 B 集文献和患者清单校验。不能只因召回率较高就进入最终提交。

## 复现与归档

- `remote_runbook.md`：环境验收、短训练和正式训练命令。
- `environment_acceptance.json`：独立验收与完成证据。
- `fold0/manifest.json`：固定模型、参数、文献划分、数据及代码哈希。
- `fold0/training.jsonl`、`fold0/summary.json`：逐步日志、各轮验证及完成状态。
- `fold0/best_validation_predictions.jsonl`：用于复核与阈值诊断的原始保存预测。
- `tokenization_audit.json`：按实际宽度 64 汇总的完整训练标注覆盖审计。
- `umls_validation.json`：术语资源验证边界。
- `b_target_validation.json`：服务器 B 原文件哈希、文献顺序、患者数与数据角色。
- `resource_snapshot.json`：最新资源、实际路径及 best/last 检查点 SHA-256。

检查点留在服务器，最佳权重约 438.8 MB，完整续训状态约 1.31 GB。本机仅归档代码与小型诊断结果。新增多个折或保留更多优化器状态前，需继续检查磁盘余量。

本次未执行 Git commit 或 push。训练代码、锁定环境、测试和本报告构成可独立提交的阶段成果；建议提交信息：`Add reproducible BiomedBERT span training and first-fold diagnostics`。
