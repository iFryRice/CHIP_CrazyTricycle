# PatientPheX CPU 基线报告

选择方案：`dictionary+nearest`。训练文献 80 篇，A 榜文献 20 篇、患者 53 人。

## 离线验证

按文献进行固定 5 折划分，单患者/多患者分层；每折词典、消歧统计和分类器仅使用该折训练文献。预测前移除答案字段。固定 HPO 本体可用于全部折。

以下是用于选择方案的交叉验证分数，不能视为独立测试或 A 榜成绩。官方评测脚本未提供，本地实现依据项目内公开说明，存在正文范围及特殊实体处理差异。

| 指标 | Precision | Recall | F1 |
|---|---:|---:|---:|
| 实体提及 | 0.7538 | 0.6309 | 0.6869 |
| 文档概念 | 0.8648 | 0.6373 | 0.7338 |
| 患者关联 Micro | 0.6594 | 0.5537 | 0.6019 |
| 患者关联 Macro | 0.6057 | 0.5402 | 0.5363 |

四项 F1 等权总分：**0.6397**。

| 配置 | 折外汇总分 | 各折均值 ± 标准差 |
|---|---:|---:|
| dictionary+nearest | 0.6397 | 0.6407 ± 0.0102 |
| dictionary+learned | 0.6300 | 0.6308 ± 0.0173 |
| learned+nearest | 0.6344 | 0.6346 ± 0.0149 |
| learned+learned | 0.6354 | 0.6354 ± 0.0132 |

## 正式预测与校验

选定配置用全部 80 篇训练文献重建后预测 A 榜。输出 1335 个实体、480 条患者概念关联；文件 155,621 字节，低于 100,000,000 字节。

已检查完整文献覆盖、主键唯一、患者引用、原文字符跨度、HPO 分支、JSONL 编码与文件大小。输入文件 SHA256 和环境版本见 `metrics.json`；逐项检查见 `submission_validation.json`。

## 适用范围

本版仅用 CPU、训练标注和随数据提供的 HPO 本体，无外部大模型推理，无平台提交。主要局限是未见表达的召回、精确边界、词义歧义及跨段患者共指。

## 复现

```powershell
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -m patientphex run
uv run python -m patientphex validate --input submissions/patientphex_a.jsonl
```

方法实现参考：[scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)；隔离训练与验证参照 [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)。
