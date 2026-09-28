# PatientPheX B 榜训练与预测

面向 CHIP 2026，从文献中识别 HPO 表型，并将表型关联到给定患者。项目保留原 CPU 基线，新增 BiomedBERT 跨度模型、指定版 HPO/UMLS 映射和患者关联训练，不调用外部大模型或付费 API。

**当前阶段为 B 榜。** 新模型已使用官方 80 篇标注文献完成 5 轮全量训练，推理目标为 `PatientPheX-V1-B/PatientPheX-V1-B.jsonl`。B 集只用于推理，不用于选轮次或阈值。

## 当前 B 榜候选

`submissions/patientphex_b.jsonl` 覆盖 100 篇文献、244 名患者，已通过服务器及本机严格校验，尚未上传比赛平台。完整结果、检查点与复现记录见 [B 榜报告](reports/b_supervised/report.md)。

**该候选尚未超过原 CPU 基线。** 同一批 80 篇文献的五折开发近似总分为 0.6014，冻结 CPU 对照为 0.6397；这些不是独立测试分数或 B 榜成绩。主要问题是实体误报增加及已覆盖概念的患者关联漏检。

在本机只读复验 B 文件，不会重训：

```powershell
uv run --no-sync python -c "from patientphex.data import read_jsonl; from patientphex.ontology import Ontology; from patientphex.validation import validate_submission; print(validate_submission('submissions/patientphex_b.jsonl', read_jsonl('PatientPheX-V1-B/PatientPheX-V1-B.jsonl'), Ontology('PatientPheX-V1-A/hp.obo')))"
```

## 历史 A 榜 CPU 基线

运行完成后的正式文件是 `submissions/patientphex_a.jsonl`，对应 A 榜全部 20 篇文献。上传此文件，不要上传 `submit_pred_ex.jsonl` 或训练集。程序仅生成文件，不会向赛事平台提交。

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
| `submissions/patientphex_a.jsonl` | 唯一正式 A 榜提交文件 |
| `reports/cpu_baseline/` | 划分、折外预测、指标、校验、输入/代码 SHA256 |
| `pyproject.toml`, `uv.lock` | 依赖声明与锁定版本 |

报告和提交文件保留在 Git 可见路径；本地环境与缓存忽略。使用 Git 管理历史，不以 `v1/v2` 文件副本区分版本。

## BiomedBERT 实体跨度训练

新增训练保留嵌套、交叉和否定实体，使用原文偏移及已有文献级五折划分。GPU 依赖位于可选 `training` extra，默认 CPU 依赖保持独立。模型、依赖和数据的版本及验证边界见 `reports/span_ner/remote_runbook.md`、`environment_ledger.md` 和 `umls_validation.json`。

远端项目实际存放在 `10.253.27.177:/home/dcf/chip2026/`，UMLS/HPO 存放在 `/home/dcf/umls/`。使用项目 `.venv` 训练；服务器 SQLite 读取使用 `/usr/bin/python3`。环境验收、短训练、五折开发评估及最终 B 推理均已完成。完整流程已接入 HPO/UMLS 映射和患者关联；不能将跨度 F1 当作比赛总分。

首轮 fold 0 的历史诊断保留于 `reports/span_ner/initial_training_report.md`。最终使用的冻结配置在 `experiments/span_ner/selected_pipeline.json`，完整五折结果在 `reports/b_supervised/development_summary.json`。`scripts/complete_b_pipeline.sh` 记录本次接续过程，已有输出会受到保护，不能当作可重复覆盖运行的命令。
