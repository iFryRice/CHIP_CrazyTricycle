# CHIP 2026 PatientPheX — 会话交接

生成日期：2026-09-28（Asia/Shanghai）。项目：`D:\Projects\20260927_CHIP2026`。
用户明确调用 `$handoff`，附带参数 `ff`。参数含义未解释，不要推断为执行、微调或其他授权。当前任务是生成交接；不是启动下一轮训练。

## 接手时最先知道

- A集融合文件已交付，用户报告官方总分 **0.6267**。不得再用本地验证分数0.640983作为官方成绩或对外预测。
- 用户愿意为下一轮优化投入 **1–2天**。已给出实体抽取、HPO映射、患者关联监督训练方案，但上一条答复明确等待确认；用户随后只调用handoff，**尚未确认启动新训练**。
- 用户要求不要重复已经完成的本地CPU路线。现有CPU/GPU/RAG/融合成果均已完成；保留它们供对照，不覆盖、不重新跑整套旧实验。
- UMLS暂不可用；用户表示之后提供权限。当前提交没有调用UMLS API或使用新增UMLS词表，不把建议写成已执行方法。
- **新发现：B集目录已出现。** 本次只读预检确认存在 `D:\Projects\20260927_CHIP2026\PatientPheX-V1-B\PatientPheX-V1-B.jsonl`，2,027,053 bytes。尚未检查记录数、schema、患者覆盖、标注是否为空或生成预测。`AGENTS.md`仍称B集不存在，已过时；不要自动改治理文件或复用A集预测作为B集结果。
- 最初30/50分钟截止约束属于上一轮紧急交付，不要把它继承为本轮新训练时限。

## 尚未写入旧实验报告的官方反馈

按用户给出的顺序，融合版官方四项：

| 指标 | 官方A集 |
|---|---:|
| 实体提及F1 | 0.6646 |
| 文档F1 | 0.7326 |
| 关联Micro-F1 | 0.5967 |
| 关联Macro-F1 | 0.5128 |

均值0.626675，四舍五入0.6267。CPU版同一A集官方成绩仍未提供，因此不能认定融合使官方成绩升高或降低。

用户贴出Agriv／中国科学院文献情报中心的五个数值：0.8071、0.7891、0.8531、0.7783、0.8077。算术上首个应为总分，后四项均值0.80705。表头未提供，后四项按上述指标顺序解释是推断；天池实时榜单未能通过web工具访问，不声称独立验证了当前名次或对方技术方案。

只读核验发现，A集融合与冻结CPU相比，entities完全不变，仅在PMC 6133598新增24条关联、PMC 6358355新增10条关联，其余18篇不变。因此融合关联规则不影响两项实体指标。已交付文件与项目源文件字节一致。

## 已有成果：引用这些文件，不重建或复制报告正文

- 完整方法、实验比较、运行耗时、局限：`D:\Projects\20260927_CHIP2026\reports\gpu_comparison\final_report.md`。
- 参数、校验与交付清单：`D:\Projects\20260927_CHIP2026\reports\gpu_comparison\final_manifest.json`。
- 12种融合比较、80篇/72篇及单/多患者分组：`D:\Projects\20260927_CHIP2026\reports\fusion_search\comparison.json`。
- CPU方法与环境：`D:\Projects\20260927_CHIP2026\reports\cpu_baseline\report.md`、同目录`metrics.json`，以及项目`pyproject.toml`、`uv.lock`。
- 冻结输入：`D:\Projects\20260927_CHIP2026\work\gpu_route\cpu_a_input.jsonl`、`cpu_oof_input.jsonl`、`cpu_metrics_input.json`。早期learned+nearest结果已过时；冻结基线实际是dictionary+nearest。
- GPU、RAG、融合实现：项目`scripts\gpu_refine.py`、`scripts\build_rag_contexts.py`、`scripts\fuse_predictions.py`；诊断脚本位于`work\gpu_route\`。
- 最佳A集项目源：`D:\Projects\20260927_CHIP2026\submissions\patientphex_a_cpu_gpu_union_multi.jsonl`。
- 用户可见交付：`D:\Cache\Codex\2026-09-27\cha-xu\outputs\patientphex_a_fusion.jsonl`；同目录`patientphex_comparison_report.md`、`patientphex_manifest.json`。该目录另有CPU、普通GPU和RAG的分别导出，提交推荐的是fusion。
- 已交付fusion SHA256：`66a1ddb7910dd2f06d35470cd1b65e2df65054819bad730350b5dd0ef32f3b50`。规则为单患者保留CPU、多患者取CPU与普通GPU关联并集。共20篇、53患者、1335实体、514关联；格式校验已通过。

旧报告是在用户反馈官方成绩之前形成的。阅读时把其所有分数视为本地实验结果，不把它当作最新官方成绩记录。

## 新诊断与下一轮方案（未执行）

本轮只读重核冻结80篇OOF：2171条真实患者关联仅1465条在实体候选中，遗漏706条（32.52%）；其中691条遗漏的概念在同文献正向gold实体中已有标注。其余14条只见于否定实体、1条未见对应实体，属于需分析的标注口径问题，不应擅自修改gold。瓶颈指向实体提名/归一化漏检，而非仅提示词。

相对用户所贴队伍，关联两项贡献约66%的总分差距；但即使关联达到该队水平，固定当前实体两项，总分也只有0.7458。要同时改实体与关系。未知对方方法，不能据分数推断其模型。

推荐1–2天计划：
1. 前2–3小时：先核查新B集、任务阶段与可提交时间；固定文献分组和错误类别；确认GPU、磁盘和环境；用短训练测吞吐及峰值显存，再估算作业时间。
2. 第一天主线：在比赛80篇训练标注上训练BiomedBERT跨度识别（重叠窗口、原文offset还原、否定、复合和-1）；允许提出词典外原文跨度。HPO用指定2026-06-23版本HP:0000118分支，名称/同义词/定义检索，再用SapBERT及上下文重排比较。词典保留作补充，不再冻结全部实体。PhenoTagger是官方基线参考，若接入须核对HPO版本与训练来源。
3. 同步准备患者关联监督：患者的多处mention、概念的原文证据段落、章节和患者—概念标签；先训练小模型对照，再测试Qwen3-8B LoRA/QLoRA。重点是其他患者、家属、否定、背景与跨段指代困难负例，平衡患者采样，避免只优化大患者集合。
4. 第二天：单模块对照后再组合；少量优胜方案做文献级完整验证和必要的稳定性检查；最后生成新目标集结果、校验和技术报告。不要继续盲搜大量融合规则。

技术约束：gold仅有患者—概念集合，没有逐条人工证据链，不能把自动检索段落或LLM生成理由当人工证据标签。未标注关联未必都是可靠负例。关系训练应接触折外预测候选，避免只用完美gold实体造成训练/推理分布差异。参数总量也保守控制在10B以内，实际权重版本/参数量必须核验；Qwen3-8B模型卡声明8.2B。目前没有进行任何SFT、LoRA、新NER训练或UMLS增强。

时间表是工作预算，不是已测完成时间；不承诺达到0.80。

## 验证边界与规则来源

- 80篇已用于CPU、提示、RAG与12种融合反复选型；排除8篇pilot的72篇仍不是独立测试。新本地结果不能包装为独立泛化成绩。
- 文献为分组单位，窗口、患者、别名和训练示例均跟随文献。原5折OOF的RAG必须排除目标所属整个16篇fold，只检索其余64篇；A/B只能使用允许的训练资源，不用目标答案。
- 本地评分四项等权、逐患者macro等实现与文字规则相符，但正文范围、文档级-1等仍有未明确口径；无证据断定0.6267源于代码算错。详情参考`patientphex\evaluation.py`的APPROXIMATIONS。
- 最新官方仓库比本地保存说明多一项重要约束：**不允许使用外部人工标注表型数据集**；外部公开代码、工具和知识库可用，须注明来源。不要直接拿其他人工标注表型语料来微调，也不要照抄外部仓库以其人工dev集选模型的命令。
- 官方说明还区分A开发榜与B最终评测，须接手时核对实际阶段。参考：https://github.com/DUTIR-BioNLP/PatientPheX/ 。当前仓库未提供官方评分脚本。
- 参考模型/方法：https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext 、https://aclanthology.org/2021.naacl-main.334/ 、https://huggingface.co/Qwen/Qwen3-8B 、https://github.com/ncbi-nlp/PhenoTagger 。这些是建议及已查资料，不代表已下载/训练。

## 运行与权限注意

- 用户最新贴出的AGENTS.md替换了旧版全局指令：中文沟通；代码/注释英文；先具体方案并获得确认再执行；只读分析可直接做。新1–2天训练方案尚待确认，handoff本身已获明确授权。
- Git仓库存在，但大量项目文件仍untracked；不stage、commit、push，不撤销用户或另一CPU会话的修改，尤其保留`submissions_dcf_CPU`。未产生本任务commit。
- Python使用项目内`.venv`和uv、pyproject/uv.lock，禁止改全局环境。现有项目Python3.12。远程新环境也在任务项目中隔离，需依照确认方案安排。
- MEMORY.md、FILE_MANAGEMENT.md当前仍缺失；建议后续用户执行/init，不自行初始化，不阻塞当前只读交接。不要自动写MEMORY。
- 历史远程信息（均为2026-09-27，必须刷新）：209服务器有2×A10040GB并运行既有Qwen3-8B vLLM服务；177有4×V10032GB。既有共享服务未被停止。两机当时均有他人GPU作业，209磁盘余量较紧，不能把旧空闲量当作当前可用资源。
- 209既有SSH配置可用。曾使用localhost:18089隧道转发209:8089，本任务隧道已关闭。不要直接重跑依赖隧道的命令并假定服务可达；不要重启共享vLLM。
- 177认证曾依赖用户临时提供的凭据，最后状态不可复用。交接不保存任何凭据；需要时通过安全方式重新授权，不打印/保存剪贴板或密钥。
- 209此前发现高CPU可疑root进程及异常cron/服务线索，仅报告、未清理。此信息已陈旧，不把它直接断定为入侵或自行停进程。
- 本会话默认sandbox曾因setup refresh错误不可用，后续限定范围的exec以require_escalated通过自动审核。下一会话按实际工具状态处理，勿自动扩大权限。
- 用户可见交付链接依当前桌面约束优先用`D:\Cache\Codex\2026-09-27\cha-xu\outputs`。本交接主文件按handoff技能保存OS临时目录，下载副本也放outputs，不在项目根生成交接副本。

## Suggested skills

按下一步实际任务选择，先读取各SKILL.md；不要因为被列出就全调用：

- `run-experiment` — `C:\Users\Fan\.codex\skills\run-experiment\SKILL.md`：确认方案后部署和运行GPU实验。
- `experiment-plan` — `C:\Users\Fan\.codex\skills\experiment-plan\SKILL.md`：把已讨论路线落实为有限候选、验收与时间预算。
- `experiment-audit` — `C:\Users\Fan\.codex\skills\experiment-audit\SKILL.md`：文献划分、检索和训练数据泄漏审计。
- `analyze-results` — `C:\Users\Fan\.codex\skills\analyze-results\SKILL.md`：分析四项指标、分组表现与稳定性。
- `monitor-experiment` — `C:\Users\Fan\.codex\skills\monitor-experiment\SKILL.md`：仅在新作业实际启动后监控。
- `handoff` — `C:\Users\Fan\.codex\skills\handoff\SKILL.md`：必要时继续压缩交接。

建议下一条消息：确认已读取报告和以上官方反馈，提示B集新出现，确认下一轮要针对A调优还是直接准备B，以及用户是否执行已提出的1–2天方案；可同时做不依赖回答的只读预检。不要因ff自行启动训练。