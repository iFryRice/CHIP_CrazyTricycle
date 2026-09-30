# PatientPheX B 榜训练与预测

面向 CHIP 2026，从文献中识别 HPO 表型，并将表型关联到给定患者。项目保留原 CPU 基线，新增 BiomedBERT 跨度模型、指定版 HPO/UMLS 映射和患者关联训练。Qwen3-8B 优化试验也在 177 服务器本地运行，不调用外部推理 API。

**当前阶段为 B 榜。** 新模型已使用官方 80 篇标注文献完成 5 轮全量训练，推理目标为 `PatientPheX-V1-B/PatientPheX-V1-B.jsonl`。B 集只用于推理，不用于选轮次或阈值。

2026-09-30 实验记录：[中文版](reports/experiment_record/zh.md) / [English](reports/experiment_record/en.md)。覆盖最新代码、joint 的训练与顺序推理、两折与五折结果、B 产物、失败试验及复现限制。

## 当前 B 榜候选与官方反馈

已确认的官方 B 总分为原纯 CPU 的 **0.6152**，对应 `submissions/patientphex_b_cpu.jsonl`；见 [官方反馈](reports/official_feedback/b_cpu.json)。

**最新候选为 [`submissions/patientphex_b_joint.jsonl`](submissions/patientphex_b_joint.jsonl)**：在原上下文候选上应用 Qwen 概念含义核对，再融合全量训练上下文模型与 15 个 BiomedBERT 关联模型。覆盖 100 篇文献、244 名患者，含 6957 个实体、2207 条关联。SHA256：`7ad32cfddfc783cbb40a105fbee2bb433ddf2b6120b56a8a7861a107475d5a0c`。服务器和本机严格校验均通过，15 个模型的保存概率可精确重建最终预测，4621 条受保护实体均保持原记录。见 [B 推理报告](reports/joint_b/report.md) 和 [独立重放验证](reports/joint_b_verification/summary.json)。尚无该文件的官方分数，未自动向比赛平台提交。

原上下文候选 `submissions/patientphex_b_context.jsonl` 保留；其 7310 个实体、2301 条关联与原指纹均未改变。

相同 80 篇训练文献的五折开发总分，原 CPU 为 0.639727，CPU + UMLS 为 0.645014，语义映射为 0.668747，当前上下文关联为 **0.681916**。当前方案实体指标不变，关联微/宏 F1 均改善，四折提高、一折下降 0.001691，减少 186 条假关联、多漏 4 条真关联。完整证据与代价见 [上下文关联报告](reports/association_context/report.md) 和 [语义映射报告](reports/semantic_linking/report.md)。这些是反复用于开发的交叉验证分数，不能写作官方 B 分数，也不能直接用于推算 B 榜提升。

[BIO 两折替换试验](reports/bio_ner/report.md) 未通过扩展门槛，已停止扩展。[文档缩写试验](reports/document_abbreviations/report.md) 得到 0.683887 的开发分，增益 0.001971 未达 0.003 晋级门槛，没有生成新的 B 文件。[BIO 补充证据试验](reports/bio_corroboration/report.md) 通过两折试验后完成其余三折，完整开发分为 0.686504，但后三折平均增益仅 0.000710，未通过冻结的确认门槛；没有启动全量拟合或生成 B 文件。仍在向 0.70 的官方目标推进。

[Qwen3 精确原文提取试验](reports/llm_extraction/report.md) 完成两折对照；直接补充表型的最佳开发分为 0.674790，略低于同折基线 0.674959。[跨度支持过滤](reports/llm_corroboration/report.md) 的缩写配置平均提高 0.006051，但未达到 0.008 门槛。两项均未晋级、未生成 B 文件；下一轮转向患者归属的语义判断。

[患者归属复核试验](reports/llm_association/report.md) 的自由引文协议因引用不匹配提前停止；[原文编号协议](reports/llm_association_references/report.md) 又暴露出提示词无法保证引用数量的问题，两项均未进入准确率评分。[结构化解码](reports/llm_association_constrained/report.md) 完成 700 条任务，格式成功率为 100%，但保守策略删除 41 条假关联的同时误删 19 条真关联，两折汇总分从 0.674959 降至 0.673582，未晋级。[患者身份标记试验](reports/llm_patient_markers/report.md) 的保守策略为 0.674561，删除 42 条假关联、误删 23 条真关联，也未晋级，停止扩展直接删除关联的路线。

最新[概念含义核对试验](reports/llm_concept_review/report.md) 针对缩写歧义和过度具体化映射，核对当前 HPO 含义是否被原文支持；不判断患者归属，也不因一般讨论或否定表达删除实体。747 条任务完成，格式成功率 100%；两折汇总分 0.679213，相比同折基线增加 0.004254，平均折增益 0.004528，未达预设 0.008 门槛。删除 97 个错误实体评价单元、误删 19 个正确单元，未替换 B 文件。NO、未映射、复合 ID 和唯一精确 HPO 术语均受保护。

上一份语义映射候选 `submissions/patientphex_b_semantic.jsonl` 和纯 CPU + UMLS 版本 `submissions/patientphex_b_cpu_umls.jsonl` 均保留，后者报告见 [UMLS 增量优化](reports/umls_augmentation/report.md)。较早的关联降权、原神经 B 文件及 A 榜文件也均保留。

本机校验：

```powershell
uv run --no-sync python -m patientphex validate --target-set B --input submissions/patientphex_b_context.jsonl --report reports/association_context/local_validation.json
uv run --no-sync python -m patientphex validate --target-set B --input submissions/patientphex_b_cpu_umls.jsonl --report reports/umls_augmentation/local_validation.json
```

## 历史 A 榜 CPU 基线

历史 CPU A 榜文件保存在 `submissions_dcf_CPU/patientphex_a.jsonl`，对应 A 榜全部 20 篇文献。下方保留的 A 榜命令默认输出到 `submissions/patientphex_a.jsonl`。程序仅生成文件，不会向赛事平台提交。

本地校验报告：`reports/cpu_baseline/submission_validation.json`；离线成绩与限制：`reports/cpu_baseline/report.md`。平台实际成绩以官方反馈为准。

## 环境与命令

在项目根目录使用 Python 3.12 和 `uv`，依赖安装在 `.venv`：

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -m patientphex run
uv run python -m patientphex validate --input submissions/patientphex_a.jsonl
```

`run` 固定种子 `20260927`，执行按文献分组、单/多患者分层的五折验证，比较预设的四个配置；按折外汇总分选择配置，再用全部 80 篇训练文献重建，预测 A 榜并严格校验。重复运行写回相同成果路径；可通过 Git 保存代码与成果历史。

## 方法与验证边界

- 实体识别：HPO 名称/精确同义词、训练别名、词元 trie、训练频次消歧、文档内缩写与否定规则；可选稀疏逻辑回归过滤候选。
- 患者关联：患者提及、句段距离、章节归属与临床语境；比较最近患者规则和稀疏逻辑回归。
- 每折所有训练派生词典和分类器只使用该折训练文献。预测前移除答案字段，不依据 A 榜结果调参。
- 字符位置均来自原始段落切片。HPO 固定为 `2026-06-23` 的 `HP:0000118` 后代；异常、分支外或废弃 ID 不直接输出。
- 离线评分依照 `官网说明.md` 实现；没有官方 scorer。正文范围、`-1` 等处理假设见报告中的 `approximations`。训练标签中的异常标识不擅自修正评分真值。
- 四配置的交叉验证用于方案选择，并非独立测试结果。主要局限为未见表述、精确边界、歧义消解与多患者跨段共指。

## 目录

| 路径 | 用途 |
|---|---|
| `PatientPheX-V1-A/` | 原始数据、提交示例和指定 HPO，本流程只读 |
| `patientphex/` | 提取、关联、评分、校验与命令行实现 |
| `tests/` | 格式、指标、原文偏移和训练隔离回归测试 |
| `submissions_dcf_CPU/patientphex_a.jsonl` | 历史 CPU A 榜归档文件 |
| `reports/cpu_baseline/` | 划分、折外预测、指标、校验、输入/代码 SHA256 |
| `pyproject.toml`, `uv.lock` | 依赖声明与锁定版本 |

报告和提交文件保留在 Git 可见路径；本地环境与缓存忽略。使用 Git 管理历史，不以 `v1/v2` 文件副本区分版本。

## BiomedBERT 实体跨度训练

新增训练保留嵌套、交叉和否定实体，使用原文偏移及已有文献级五折划分。GPU 依赖位于可选 `training` extra，默认 CPU 依赖保持独立。模型、依赖和数据的版本及验证边界见 `reports/span_ner/remote_runbook.md`、`environment_ledger.md` 和 `umls_validation.json`。

远端项目实际存放在 `10.253.27.177:/home/dcf/chip2026/`，UMLS/HPO 存放在 `/home/dcf/umls/`。使用项目 `.venv` 训练；服务器 SQLite 读取使用 `/usr/bin/python3`。环境验收、短训练、五折开发评估及最终 B 推理均已完成。完整流程已接入 HPO/UMLS 映射和患者关联；不能将跨度 F1 当作比赛总分。

首轮 fold 0 的历史诊断保留于 `reports/span_ner/initial_training_report.md`。最终使用的冻结配置在 `experiments/span_ner/selected_pipeline.json`，完整五折结果在 `reports/b_supervised/development_summary.json`。`scripts/complete_b_pipeline.sh` 记录本次接续过程，已有输出会受到保护，不能当作可重复覆盖运行的命令。

下一轮[监督式关联文本分类](experiments/relation_text/plan.json) 使用现有 BiomedBERT 学习患者归属和比赛标注口径。固定三轮训练，比较单模型与两种预先确定的融合比例；训练候选严格来自每折训练文献的内层交叉预测。服务器 144 项测试全部通过；修正实际词表没有保留词元的问题后，20 步真实训练与推理检查通过。两折训练已完成：25% 文本模型 + 75% 上下文模型总分 0.685286，平均折增益 0.010512，通过预设门槛；固定该比例完成剩余三折确认：全五折 0.687068，四折提升，但剩余三折平均增益 0.001960、bootstrap 下界 -0.000191，未通过全部门槛，未生成 B 文件。纯文本模型和 50% 融合均未晋级。

后续[三随机种子关联集成](experiments/relation_seed_ensemble/plan.json) 固定使用 20260929、20260930、20261001 的等权概率平均，新增十次折训练；不挑选种子、不调整融合比例和门槛。十次新增训练均完成；三种子五折总分 0.687283，后面三折平均增益仅 0.000986，未通过全部门槛。只筛除关联的 CPU 消融为 0.684913，也未晋级。所有训练队列已结束，原 B 候选保持不变。汇总见 [本轮优化状态](reports/optimization_status.md)。

最新[联合流水线试验](reports/joint_review/report.md) 将概念含义核对与三种子关联融合顺序执行，受影响的文本输入重新推理，其他输入经 token 一致性校验复用。[全五折确认](reports/joint_confirmation/report.md) 得到 **0.690770**，相比上下文基线提高 0.008854，后三折平均增益 0.003864，全部预设门槛通过。154 项服务器回归通过。按[冻结的 B 推理计划](experiments/joint_b/plan.json) 完成了 2689 条概念核对与全部 15 个文本模型的推理，严格校验后生成 `submissions/patientphex_b_joint.jsonl`。开发分数仍不是官方 B 分数。

后续[受限重叠候选筛选](reports/nested_span_screen/report.md) 在两折补回 50 个真实体，却新增 311 个假实体，实体 F1 降低，已停止扩展；未影响联合 B 文件。剩余召回与患者归属问题见 [诊断报告](reports/entity_residuals/joint_report.md)。
