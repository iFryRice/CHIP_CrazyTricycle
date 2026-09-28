# PatientPheX 最终比较与交付报告

生成时间：2026-09-27T23:53:23+08:00。

推荐候选采用统一规则：**单患者文献完整保留CPU；多患者文献保留CPU关联，并补充普通GPU关联（CPU ∪ GPU）**。最终融合不使用RAG输出。
全80篇本地诊断总分由0.639727提高至0.640983；排除原8篇pilot的72篇由0.633688提高至0.635222。提升较小。**这是12规则选择后的本地诊断，不是独立测试，不保证官方A榜改善。**
本次没有使用UMLS，没有新增训练CPU或大模型权重。

## 交付与复验

推荐提交：`patientphex_a_fusion.jsonl`；同时保留CPU、普通GPU和RAG结果。
已从冻结CPU和完整普通GPU输出独立重算融合，逐条等于提交结果。严格校验通过：20篇、53患者、1335实体、514关联，156,077字节，warnings为空。
实体、PMID、文献顺序不变；全部单患者记录保留CPU；多患者执行同一集合规则。没有按目标gold逐篇选择答案。
SHA256：`66a1ddb7910dd2f06d35470cd1b65e2df65054819bad730350b5dd0ef32f3b50`。
A集没有gold，本次仅验证结构、覆盖、跨度、HPO和融合规则；未计算A集准确率。

## 四项F1与总分

### 全部80篇折外预测

| 路线 | 实体F1 | 文档F1 | 关联Micro F1 | 关联Macro F1 | 总分 |
|---|---:|---:|---:|---:|---:|
| CPU | 0.686909 | 0.733802 | 0.601903 | 0.536294 | 0.639727 |
| 普通GPU | 0.686909 | 0.733802 | 0.603261 | 0.519397 | 0.635842 |
| RAG | 0.686909 | 0.733802 | 0.600548 | 0.537562 | 0.639705 |
| 最终融合 | 0.686909 | 0.733802 | 0.605005 | 0.538216 | 0.640983 |

### 非pilot72篇

| 路线 | 实体F1 | 文档F1 | 关联Micro F1 | 关联Macro F1 | 总分 |
|---|---:|---:|---:|---:|---:|
| CPU | 0.682136 | 0.728996 | 0.590210 | 0.533410 | 0.633688 |
| 普通GPU | 0.682136 | 0.728996 | 0.593999 | 0.516170 | 0.630325 |
| RAG | 0.682136 | 0.728996 | 0.588923 | 0.534940 | 0.633749 |
| 最终融合 | 0.682136 | 0.728996 | 0.594092 | 0.535664 | 0.635222 |

72篇仍属于反复用于诊断和选型的训练资料，不能视为新独立测试集。实体被冻结，所以实体和文档F1保持不变。

## 12种确定性融合候选

| 规则 | 仅多患者融合 | 全80总分 | 相对CPU | 非pilot72总分 | 相对CPU |
|---|---|---:|---:|---:|---:|
| CPU∪GPU | 是 | 0.640983 | +0.001256 | 0.635222 | +0.001534 |
| 三路多数票 | 是 | 0.640742 | +0.001014 | 0.634896 | +0.001207 |
| CPU∪(GPU∩RAG) | 是 | 0.640685 | +0.000958 | 0.634834 | +0.001146 |
| CPU∪GPU | 否 | 0.640633 | +0.000906 | 0.634842 | +0.001154 |
| CPU∪(GPU∩RAG) | 否 | 0.640340 | +0.000613 | 0.634460 | +0.000772 |
| 三路多数票 | 否 | 0.640180 | +0.000453 | 0.634441 | +0.000753 |
| CPU∪RAG | 是 | 0.639965 | +0.000238 | 0.634047 | +0.000359 |
| CPU∪RAG | 否 | 0.639760 | +0.000033 | 0.633673 | -0.000015 |
| CPU∩RAG | 是 | 0.639691 | -0.000037 | 0.633631 | -0.000057 |
| CPU∩RAG | 否 | 0.639666 | -0.000061 | 0.633762 | +0.000074 |
| CPU∩GPU | 是 | 0.635740 | -0.003988 | 0.629255 | -0.004433 |
| CPU∩GPU | 否 | 0.634814 | -0.004913 | 0.629045 | -0.004643 |

选择依据为全80总分；72篇用于一致性检查。最佳规则在两个口径中均排名第一。

## 分组与取舍

| 范围 | 患者组 | CPU总分 | 融合总分 | 差值 | 改善/恶化/同分篇数 |
|---|---|---:|---:|---:|---|
| 全80 | 全部 | 0.639727 | 0.640983 | +0.001256 | 6/5/69 |
| 全80 | 单患者 | 0.666774 | 0.666774 | +0.000000 | 0/0/36 |
| 全80 | 多患者 | 0.633075 | 0.634848 | +0.001773 | 6/5/33 |
| 非pilot72 | 全部 | 0.633688 | 0.635222 | +0.001534 | 6/4/62 |
| 非pilot72 | 单患者 | 0.653631 | 0.653631 | +0.000000 | 0/0/32 |
| 非pilot72 | 多患者 | 0.629986 | 0.632101 | +0.002115 | 6/4/30 |

全80中融合增加31条正确关联，也增加51条错误关联。召回提高而精确率下降，Micro和Macro F1均小幅改善，但并非每篇文章改善。

## 实测运行耗时

| 阶段 | 最后调用耗时 | 新请求/缓存说明 |
|---|---:|---|
| CPU完整5折 | 87.340秒 | 完整CPU流水线 |
| 普通GPU pilot | 40.287秒 | 8个新请求，0篇已完成缓存 |
| 普通GPU全80续跑 | 831.260秒 | 72个新请求，8篇已完成缓存 |
| 普通GPU A集最终修复 | 12.055秒 | 1个新请求，19篇已完成缓存 |
| RAG pilot | 101.732秒 | 8个新请求，0篇已完成缓存 |
| RAG全80续跑 | 879.817秒 | 72个新请求，8篇已完成缓存 |
| RAG A集 | 299.801秒 | 20个新请求，0篇已完成缓存 |

普通GPU/RAG全80续跑各复用8篇pilot。普通GPU A集最后summary只记录1篇严格schema修复，12.055秒不是20篇全程时间。阶段可能并行、共享服务，耗时不能直接相加解释为整个任务墙钟时间。最终融合只做本地集合运算，没有新增推理。

## 方法与复现说明

以下纳入本次methods.md的方法说明，并更新最终完成状态。完整12候选和分组四项指标见项目reports/fusion_search/comparison.json。

# PatientPheX：CPU 与 GPU 关联修正方法

本实验以已完成的 CPU 预测为冻结输入，对比 Qwen3-8B 普通提示和检索增强提示（RAG）。GPU 仅修改 `association`，保留 `entities` 的内容、顺序、字符偏移及否定标记；不增加 CPU 未提出的表型候选。本节描述方法和复现条件，不报告成绩。

## 数据、资源与模型来源

使用项目随附的 80 篇训练文献、20 篇无标签 A 集文献和 `hp.obo`。HPO 版本为 `2026-06-23`，有效表型限制在 `HP:0000118` 分支。原始文档、JSON 字段及字符偏移不修改。训练与 A 集文献编号互不重叠，不使用 A 集答案。

GPU 调用 `10.249.40.209` 上已存在的 vLLM 服务，远端地址为 `http://127.0.0.1:8089/v1`，模型请求参数为 `Qwen3-8B`。官方 [Qwen/Qwen3-8B 模型卡](https://huggingface.co/Qwen/Qwen3-8B) 标明模型共 8.2B 参数，名义规模小于 10B；本实验使用外部预训练模型，未执行 SFT、LoRA 或其他大模型权重训练。已确认服务标识和调用模型名，但未取得该服务实际权重的 `config.json`、版本提交或权重 SHA256，因此不能据此声称已验证部署权重与官方发行版完全一致，也不能以本地脚本哈希代替权重校验。

服务器选择以截止前可执行性为依据。209 配备两张 A100 40GB，查询时 GPU1 约余 26.7 GiB，已有可用 Qwen 服务。177 配备四张 V100 32GB、40 CPU 线程、约 231 GiB 可用内存及 135GB 可用根分区空间；一次采样中 GPU0–3 利用率分别为 100%、99%、78%、10%，各余约 23.7–26.2 GiB 显存。177 的资源有扩展价值，但未完成其现有模型列表核验；本次 GPU 推理使用 209。上述为瞬时采样，不代表独占资源或持续可用容量。

## CPU 基线与冻结输入

CPU 代码比较词典/学习型实体抽取与最近患者/学习型关联的四种组合，学习型组件采用传统机器学习，包含 scikit-learn LogisticRegression。固定 5 折按文献划分，单患者/多患者分层，种子为 `20260927`；每折 64 篇拟合、16 篇验证。最终 CPU 配置为 `dictionary+nearest`。选型后使用全部 80 篇训练资料构建 A 集预测。

GPU 使用以下冻结快照，不在推理期间从其他运行结果动态替换输入：

- `D:\Projects\20260927_CHIP2026\work\gpu_route\cpu_oof_input.jsonl`：80 篇折外预测。
- `D:\Projects\20260927_CHIP2026\work\gpu_route\cpu_a_input.jsonl`：20 篇 A 集预测。
- `D:\Projects\20260927_CHIP2026\reports\cpu_baseline\split.json`：固定折分。

这两份预测与训练文档、A 集文档、HPO、split 的 SHA256 均记录于 `D:\Projects\20260927_CHIP2026\work\gpu_route\rag_contexts.json` 的 `metadata.source_sha256`。split SHA256 为 `2384f83aa16e1c9d63313f2e2741e844f5ff667d6419ece504332472f9c31aff`。

## 普通提示、RAG 与防泄漏

普通提示包含目标文章原文、已提供的患者编号和 mention、CPU 候选表型及暂定关联。候选由 CPU 非 `NO` 实体生成；分号分隔的 HPO 标识按各个 ID 处理，未映射实体使用原文字符串。模型必须覆盖全部给定患者，且只能选择精确匹配的候选值。提示要求排除否定、泛化疾病特征、文献案例及不属于该患者的家属描述。目标文章的 gold `entities` 和 gold `association` 不进入提示。

RAG 在相同目标输入之上加入 HPO 正式名、每条最多 300 字符的定义、最多 3 条同义词，以及最多 2 篇相似训练文章的简短示例。检索采用原始 `full_text` 的 TF-IDF 余弦相似度：英文停用词、小写、sublinear TF、最多 40,000 特征，默认词级 unigram。训练目标必须排除自身所在的整个 16 篇 held-out fold，向量器拟合和检索候选均仅来自其余 64 篇；A 集使用全部 80 篇训练文章。相似度相同时按文献编号稳定排序。

仅选纯文本检索前两篇。某篇若没有与目标 CPU 候选重合、且带非否定实体证据的训练 gold 关联，则不提供其示例，也不以更低排名替补。每个示例最多给出两条真实训练患者—表型关联、附近患者 mention 及原文窗口，总计不超过 2,500 字符。训练 gold 关联可供示例使用；窗口仅说明原文措辞，并非另行标注的成对关系证据。目标 gold 不用于查询、排序、候选选择或示例筛选。检索示例不扩充目标候选，也不能替代目标文章证据。

上下文覆盖 100 篇目标，共生成 149 个示例。每篇保存 `excluded_fold_ids`、纯文本前两名 `ranked_retrieval_ids`、实际示例 `retrieved_ids` 和 `eligible_training_ids_hash`；后两者所对应的文章均不得属于目标或其排除 fold。GPU 脚本再校验候选边界和示例编号。检索上下文参与请求缓存 identity，避免将普通提示结果误当作 RAG 结果复用。

## 推理、格式校验与审计

本次请求使用 `temperature=0`、`chat_template_kwargs.enable_thinking=false`、`max_tokens=4000` 和 `max_prompt_chars=70000`。后者是包括系统提示在内的字符预算，不是模型 token 上下文上限。采用全文输入；每篇实际是否完整以审计中的 `prompt_report.source_is_excerpted`、`covered_source_characters` 和 `source_characters` 为准。普通提示 80 篇验证、20 篇 A 集的记录均显示未截断；RAG 80篇验证及20篇A集已逐篇核验完成，均未截断，最终全部使用GPU结果。

普通提示初始使用 `response_format=json_object`。脚本还支持 `--structured-output`，向服务提交包含精确患者键、候选枚举及 `additionalProperties=false` 的严格 JSON Schema；RAG 路线使用该模式。所有输出仍经过本地患者覆盖、候选合法性、重复项及 JSON 校验。A 集普通提示的文献 `6358355` 首次出现非法 HPO 编号，原响应和错误保存在 `previous_attempts`，随后仅对该文献使用严格 schema 再推理并校验成功；不是手工修改答案。

每篇 audit 保存目标编号、模型名、提示、覆盖统计、返回内容、token 用量、耗时、最终关联及错误。`source=gpu` 表示成功使用模型输出；`cpu_fallback`、`cpu_not_requested` 或 `cpu_pending` 明确表示该篇仍保留 CPU 关联。完成性以 summary 的 `all_documents_refined` 和逐篇状态核对，不把文件存在视为全部 GPU 成功。

旧版普通提示 summary 中的 `source_gold_annotations_used=false` 表示未向模型提供目标文章的答案字段。新版用 `target_gold_annotations_used=false` 单独记录这一点，并以 `retrieval_training_annotations_used=true` 表明 RAG 使用了排除 fold 以外的训练标注；两者应同时披露，不能将 RAG 描述为完全未使用 gold。公开提供的患者 mention 仍作为任务输入使用。

## 代码、复现与评估边界

本地环境为 Python 3.12.13、scikit-learn 1.9.1，由项目 `.venv`、`pyproject.toml` 和 `uv.lock` 管理。当前仓库没有可引用的 HEAD 提交；未提交、未推送。CPU 代码哈希见 `D:\Projects\20260927_CHIP2026\reports\cpu_baseline\metrics.json`，RAG 构建脚本哈希见上下文 metadata；它们记录代码和输入来源，不保证跨硬件服务逐字确定性。

核心代码：

- `D:\Projects\20260927_CHIP2026\patientphex\`：CPU 抽取、关联、评估和提交校验。
- `D:\Projects\20260927_CHIP2026\scripts\gpu_refine.py`：普通/RAG 提示、服务调用、严格输出校验和逐篇审计。
- `D:\Projects\20260927_CHIP2026\scripts\build_rag_contexts.py`：HPO 知识与整 fold 排除的训练检索。
- `D:\Projects\20260927_CHIP2026\work\gpu_route\compare_results.py`：冻结 CPU 与候选结果的覆盖、实体不变性及本地指标对照。

以下命令是复现步骤，本报告撰写未重新训练 CPU。先在 PowerShell 中设置项目并安装锁定依赖；已有冻结输入应保留，不从后续实验覆盖：

```powershell
$ProjectRoot = 'D:\Projects\20260927_CHIP2026'
Set-Location -LiteralPath $ProjectRoot
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python "$ProjectRoot\scripts\build_rag_contexts.py"
```

CPU 从头复现入口为 `uv run python -m patientphex run`；它会重新生成默认 CPU 结果和报告。比较当前 GPU 结果时使用现存冻结快照，不需要重复该步骤。

在单独终端建立已授权的 SSH 隧道，不修改共享 vLLM 服务：

```powershell
ssh -N -o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -L 127.0.0.1:18089:127.0.0.1:8089 dcf@10.249.40.209
```

本次任务结束会关闭自己的隧道，复现时需重新建立。下例复现普通提示的 80 篇验证推理：

```powershell
uv run python "$ProjectRoot\scripts\gpu_refine.py" --base-url http://127.0.0.1:18089/v1 --model Qwen3-8B --input "$ProjectRoot\work\gpu_route\cpu_oof_input.jsonl" --data "$ProjectRoot\PatientPheX-V1-A\PatientPheX-train.jsonl" --output "$ProjectRoot\work\gpu_route\full_oof_gpu.jsonl" --report-dir "$ProjectRoot\reports\gpu_validation" --max-tokens 4000 --max-prompt-chars 70000 --fail-on-fallback
```

RAG 验证使用相同输入和参数，输出与报告目录分别改为 `D:\Projects\20260927_CHIP2026\work\gpu_route\full_oof_rag.jsonl`、`D:\Projects\20260927_CHIP2026\reports\gpu_rag_validation`，并添加：

```powershell
--context-file "$ProjectRoot\work\gpu_route\rag_contexts.json" --structured-output
```

A 集推理将 `--input` 改为 `D:\Projects\20260927_CHIP2026\work\gpu_route\cpu_a_input.jsonl`，`--data` 改为 `D:\Projects\20260927_CHIP2026\PatientPheX-V1-A\PatientPheX-A.jsonl`。普通/RAG 输出分别为 `D:\Projects\20260927_CHIP2026\submissions\patientphex_a_gpu.jsonl`、`D:\Projects\20260927_CHIP2026\submissions\patientphex_a_rag.jsonl`；对应报告目录为 `D:\Projects\20260927_CHIP2026\reports\gpu_a`、`D:\Projects\20260927_CHIP2026\reports\gpu_rag_a`。默认恢复已成功的同 identity 记录；历史普通 A 集结果包含前述单例 schema 修复，因此不是所有文献使用同一种 response format。

比较验证集、校验 A 集格式的命令：

```powershell
uv run python "$ProjectRoot\work\gpu_route\compare_results.py" --baseline "$ProjectRoot\work\gpu_route\cpu_oof_input.jsonl" --candidate "$ProjectRoot\work\gpu_route\full_oof_gpu.jsonl" --output "$ProjectRoot\reports\gpu_comparison\plain_comparison.json"
uv run python "$ProjectRoot\work\gpu_route\compare_results.py" --baseline "$ProjectRoot\work\gpu_route\cpu_oof_input.jsonl" --candidate "$ProjectRoot\work\gpu_route\full_oof_rag.jsonl" --output "$ProjectRoot\reports\gpu_comparison\rag_comparison.json"
uv run python -m patientphex validate --input "$ProjectRoot\submissions\patientphex_a_gpu.jsonl" --report "$ProjectRoot\reports\gpu_a\submission_validation.json"
uv run python -m patientphex validate --input "$ProjectRoot\submissions\patientphex_a_rag.jsonl" --report "$ProjectRoot\reports\gpu_rag_a\submission_validation.json"
```

评估阶段才将候选输出与训练 gold 比较；评估 gold 不回流到单篇目标提示。比较需确认相同文档覆盖、`entities` 完全不变和提交格式有效。CPU 配置、pilot 和提示方案都参考了本地训练文献诊断，因此本地交叉验证及方案选择结果不是独立测试成绩；官方评测脚本未提供，本地计分是对公开规则的实现，A 集官方成绩未知。最终方案和分数应由完成后的统一比较报告给出。
## 确定性关联融合

进一步比较现成 CPU、普通 GPU 与 RAG 预测的集合融合，不增加模型推理调用，不重新训练 CPU 或大模型。对每篇文献的每位给定患者，分别记三路已预测表型集合为 C、G、R；同一集合规则应用于全部文献和患者，不根据目标 gold 为个别文献选择模型或规则。

| 候选规则 | 保留的患者—表型关联 |
|---|---|
| CPU 与普通 GPU 并集 | C ∪ G |
| CPU 与普通 GPU 交集 | C ∩ G |
| CPU 与 RAG 并集 | C ∪ R |
| CPU 与 RAG 交集 | C ∩ R |
| 三路多数投票 | (C ∩ G) ∪ (C ∩ R) ∪ (G ∩ R) |
| 保留 CPU 并补充两种提示共同结果 | C ∪ (G ∩ R) |

融合仅调整 `association`；`entities`、`pmid`、文献覆盖及全部患者覆盖保持不变，空关联患者仍需输出，候选不得超出冻结实体允许的表型范围。实现入口为 `D:\Projects\20260927_CHIP2026\scripts\fuse_predictions.py`，由统一 CLI 产生候选结果并进行不变性与格式校验。具体 CLI 参数以该脚本最终 `--help` 为准。

普通 GPU 和 RAG 是共享同一 Qwen3-8B 部署权重的不同提示路线，不是分别训练的两套模型；多次提示与集合运算不使模型权重参数规模相加。此说明不改变前述部署权重版本与哈希尚未核实的限制。

融合规则在 80 篇折外预测及排除原 8 篇 pilot 后的 72 篇子集上统一比较，用于选择一个整体方法。非 pilot 的 72 篇仍属于用于反复诊断和方法选择的训练文献，不能称为独立测试集；选型后应将同一规则直接应用于 20 篇 A 集，不接触 A 集答案。本节不预设哪种融合更优，也不把新增尝试本身视为性能改善。
另定义统一的 `--multi-patient-only` 适用范围开关：单患者文献直接保留 CPU 关联，多患者文献执行选定集合规则。分流仅依据任务已提供的患者数量，对所有文献采用相同条件，不读取目标 gold，也不逐篇挑选表现较好的模型。开关已实现并参与12候选比较；最终采用仅多患者执行CPU与普通GPU并集。
