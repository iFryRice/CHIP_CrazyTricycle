# PatientPheX CPU 基线报告

选择方案：`dictionary+nearest`。训练文献 80 篇，B 榜文献 100 篇、患者 244 人。

## 离线验证

按文献进行固定 5 折划分，单患者/多患者分层；每折词典、消歧统计和分类器仅使用该折训练文献。预测前移除答案字段。固定 HPO 本体可用于全部折。

以下是本次重新训练、用于选择方案的交叉验证分数，不能视为独立测试或 B 榜成绩。目标集没有本地答案，官方成绩未知；本地实现存在正文范围及特殊实体处理差异。

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

选定配置用全部 80 篇训练文献重建后预测 B 榜。输出 6486 个实体、2158 条患者概念关联；文件 752,544 字节，低于 100,000,000 字节。

已检查完整文献覆盖、主键唯一、患者引用、原文字符跨度、HPO 分支、JSONL 编码与文件大小。输入文件 SHA256 和环境版本见 `metrics.json`；逐项检查见 `submission_validation.json`。

## 适用范围

本版仅用 CPU、训练标注和随数据提供的 HPO 本体，无外部大模型推理，无平台提交。主要局限是未见表达的召回、精确边界、词义歧义及跨段患者共指。

## 复现

已有 B 输出不可覆盖；重新运行需指定空的报告目录及新的实验输出路径。

```powershell
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -m patientphex run --target-set B --target-file PatientPheX-V1-B/PatientPheX-V1-B.jsonl --data-dir PatientPheX-V1-A --report-dir reports/cpu_b_control --output submissions/patientphex_b_cpu.jsonl
uv run python -m patientphex validate --target-set B --target-file PatientPheX-V1-B/PatientPheX-V1-B.jsonl --data-dir PatientPheX-V1-A --input submissions/patientphex_b_cpu.jsonl --report reports/cpu_b_control/submission_validation.json
```

方法实现参考：[scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)；隔离训练与验证参照 [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)。
