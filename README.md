# PatientPheX CPU Baseline

面向 CHIP 2026 的本地 CPU 基线：从文献中识别 HPO 表型，并将表型关联到给定患者。只使用随项目提供的训练标注与 HPO 本体，不调用外部大模型或付费 API。

## 直接提交

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
