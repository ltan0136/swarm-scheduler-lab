# swarm-scheduler-lab

在相同候选评估与模型调用预算下，比较单 Agent、独立多 Agent、分群协作能否找到更好的 Job Shop 调度规则。本项目是 **受限调度规则搜索**：Agent 只提出具有固定字段的 JSON 权重策略，可信的本地程序执行策略，独立校验器检查调度并重新计算 makespan。CPU 批量计算每个候选的分数，不为每一道工序调用模型。

**默认后端是完全离线的 mock。mock 结果只证明实现流程可运行，不是 LLM swarm 有效性或协作优势的实证。** 没有预设哪种模式获胜；单次小规模运行也不足以得出普遍结论。

## 快速开始

已验证的环境是 Linux、Python 3.12。直接依赖包括 JobShopLib 1.7.2、OR-Tools 9.12.4544、Matplotlib 3.11.2、jsonschema 4.26.0；完整实际依赖版本记录在 `requirements-lock.txt`。OR-Tools 版本遵循 JobShopLib 的兼容范围。

在仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/swarm-scheduler-lab smoke --output results/smoke
```

`smoke` 使用 mock，执行缩小的完整流程：生成训练与留出数据、运行三种 Agent 方法和三个基线、检查调度、求解独立参考、保存原始结果并生成图和报告。无需 API key、GPU 或外部服务。

运行多个随机种子的默认 6×6 实验：

```bash
.venv/bin/swarm-scheduler-lab run \
  --backend mock --methods all --seeds 7 19 \
  --train-count 8 --test-count 8 --jobs 6 --machines 6 \
  --max-calls 12 --max-tokens 24000 --max-output-tokens 512 \
  --max-evaluations 15 --workers 1 --solver-time-limit 2 \
  --output results/demo
```

从已有 JSON 重新生成报告，不再搜索或调用模型：

```bash
.venv/bin/swarm-scheduler-lab report \
  --input results/demo/results.json --output results/demo
```

执行测试：

```bash
.venv/bin/python -m pytest -q
```

CI 使用同一套安装、测试及离线 smoke 命令，并保存 smoke 输出供检查；CI 不读取真实模型凭据。

## 实验设计

每个作业包含必须按顺序执行的工序；每道工序有固定机器和正整数工时；同一机器的时间区间不能重叠。目标是最小化 **makespan**，即最后一道工序的结束时间。

搜索时，每个候选策略只在训练集上执行。以所有训练实例的 makespan 均值作为候选目标，越小越好。搜索完成后冻结各方法根据训练分数选出的最佳策略，再分别执行训练、留出测试和 FT06 基准评估。测试集、FT06 分数及求解器参考从不进入 Agent 提示、候选选择或群间迁移。最终图表也不会根据测试分数挑选要展示的获胜方法。

### 三种搜索组织方式

| 方法 | 历史与反馈 | 默认生成预算 | 信息交换 |
| --- | --- | --- | --- |
| `single` | 一个 Agent 连续搜索，使用自身历史 | 最多 12 次 | 无 |
| `independent` | 三个群 A/B/C 各一个 Agent，各自维护历史与优胜策略 | 每群最多 4 次，共最多 12 次 | 无 |
| `cooperative` | 同样三个群，保留本群历史及其他候选 | 每群最多 4 次，共最多 12 次 | 每完成两轮，A→B、B→C、C→A 同步交换一条本群优胜策略 |

共同的初始候选集合为 A 的短工序优先、B 的剩余工作量优先、C 的固定种子随机权重。`single` 也获得相同三个初始策略。初始候选的训练评估计入总评估预算。三个模式共用模型配置、策略空间、评估器、数据和预算上限；合作模式的迁移不额外调用模型，也不清空本群候选。

非模型基线包括：

- `shortest_processing_time`：短工序优先，固定规则，无模型调用。
- `most_work_remaining`：剩余工作量较大的作业优先，固定规则，无模型调用。
- `random_search`：固定种子随机权重搜索，使用与 Agent 方法相同的候选评估上限；固定规则只需评估自身策略。

### 策略空间和确定性

模型必须返回符合 schema 的 JSON 对象，例如短工序优先策略：

```json
{
  "weights": {
    "processing_time": 1,
    "remaining_work": 0,
    "remaining_operations": 0,
    "earliest_start": 0,
    "machine_load": 0
  },
  "tie_break": "job_id"
}
```

五个权重都必须提供，值必须是有限实数且在 `[-10, 10]` 内；未知字段会被拒绝。`tie_break` 缺省为 `job_id`，白名单还包括 `shortest_processing_time` 与 `most_work_remaining`。不执行模型生成的 Python，不使用 `exec` 或 `eval`，模型不能改动评估器、数据或约束。

可信派发器在每次选择时考虑每个作业的下一道未调度工序，定义以下特征：

| 字段 | 特征 |
| --- | --- |
| `processing_time` | 当前工序工时 |
| `remaining_work` | 当前作业从当前工序起的剩余总工时 |
| `remaining_operations` | 当前作业从当前工序起的剩余工序数 |
| `earliest_start` | 作业可用时间与机器可用时间的较大值 |
| `machine_load` | 当前机器上所有作业尚未调度工序的总工时 |

每个特征除以当前候选工序中该特征的最大值；最大值为零时，归一化特征为零。归一化后取加权和，**分数越小越优先**。剩余工作量优先因此使用 `remaining_work = -1`。并列时先使用白名单规则，再按作业编号与工序编号确定顺序。派发得到的工序从满足作业顺序与机器可用性的最早时刻开始执行。这里没有启用 JobShopLib 默认的候选剪枝：规则可以先选择未来才能开工的短工序；生成的合法调度不保证无等待或最优。

### 数据划分与参考解

- **FT06**：通过 JobShopLib 的公开基准数据加载，用于公开基准与正确性检查；已知最优 makespan 为 55。这一目标值不放入运行时 Agent 提示。合法解达到 55 时，报告可标记为“匹配已知最优”。
- **生成训练集与留出测试集**：默认各 8 个 6×6 实例。每个作业按随机顺序访问每台机器一次，工时均匀取整数 `1..20`。尺寸、数量及工时上限可配置。
- **独立随机流**：训练与测试分别由 `--data-seed`（默认 `2026`）和 split 名称派生随机流；改变训练集数量不会重新抽取测试集。`--seeds 7 19` 控制搜索重复，两次重复使用相同数据，便于配对比较；要另抽一套数据，显式改变 `--data-seed`。代码拒绝两个 split 出现完全相同的生成实例。
- **OR-Tools CP-SAT**：独立构建约束模型，用明确的单实例时间限制和一个 solver worker 求解，记录状态、目标、有效下界与耗时。`OPTIMAL` 才是求解器最优证明；`FEASIBLE` 或超时只表示当前可行解与下界，不能称作真实最优值。所有求解器调度同样经过独立校验。

每份候选及最终调度必须满足：工序完整且不重复、机器与工时匹配输入、作业工序顺序正确、机器时间区间不重叠。makespan 根据实际结束时间重新计算。无效解被拒绝，不能进入候选排行榜。

## 预算与真实模型

每个方法、每个随机种子有独立预算账本；一个方法内的所有群及并发任务共享该账本。默认上限如下：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--max-calls` | 12 | 一种 Agent 方法的总生成调用上限；分群模式均分到三群 |
| `--max-tokens` | 24000 | 一种方法的总输入与输出 token 预算 |
| `--max-output-tokens` | 512 | 每次生成的输出 token 上限 |
| `--max-evaluations` | 15 | 策略在整个训练集上评估的次数上限，包含初始三个候选 |
| `--workers` | 1 | CPU 候选评估 worker 数 |
| `--solver-time-limit` | 2 | 每个实例的 CP-SAT 参考求解秒数 |

失败、无效 JSON 和重试消耗生成调用预算，没有隐藏重试。token 预算先为待发请求保留保守额度，并在返回后记录可用计量。提供方返回的实际 token 与本地估算、缺失计量会明确区分；不会把缺失值当作零。结果同时记录调用数、失败数、候选评估数、实例调度数、墙钟时间和 CPU 时间。预算是上限，模型失败、token 不足等原因可能导致未用满；阅读结果时应比较实际消耗。

### 配置 OpenAI-compatible 接口

仓库最初没有现成模型客户端，因此提供可配置的 `/chat/completions` 客户端。将凭据安全注入环境变量 `SWARM_API_KEY`；不要把密钥放进命令参数、仓库文件或结果文件。设置 `SWARM_MODEL` 为提供方实际支持的模型 ID，可选设置 `SWARM_BASE_URL` 为兼容服务的 HTTPS API 根路径（默认 `https://api.openai.com/v1`）。不要在 URL 中放凭据或查询参数。

在环境设置中安全配置这些变量后运行：

```bash
.venv/bin/swarm-scheduler-lab run \
  --backend real --methods all --seeds 7 \
  --model "$SWARM_MODEL" \
  --base-url "${SWARM_BASE_URL:-https://api.openai.com/v1}" \
  --max-calls 12 --max-tokens 24000 --max-output-tokens 512 \
  --max-evaluations 15 --workers 1 --solver-time-limit 2 \
  --output results/real
```

这条命令中三个 Agent 方法合计最多 **36 次**生成调用，三个非模型基线不调用接口；每增加一个随机种子就增加一组独立预算。默认不自动重试；需要重试时也必须计入同一硬预算。为兼容服务配置网络访问后，先运行小规模真实对照，再考虑更大预算。不要把 mock 与真实结果混合统计。

本次交付环境未发现明确配置的模型 API key、模型 ID 或兼容接口地址，**真实模型实验未运行**。真实后端实现与实际实验结果是两项不同的交付；报告中的 mock 结果不替代真实模型验证。

## 输出与复现

输出目录包含以下可检查文件：

| 文件 | 内容 |
| --- | --- |
| `results.json` | 运行配置、实际数据实例、逐候选训练反馈、最佳策略、最终调度、求解器参考和资源消耗 |
| `summary.csv` | 各方法、种子和数据划分的 makespan、合法率与资源摘要 |
| `evaluations.csv` | 按累计候选评估次数记录的搜索轨迹 |
| `convergence.png` | 最佳训练均值 makespan 随累计候选评估次数的曲线 |
| `tokens.png` | 真实模型且有可用计量时，最佳训练结果随累计 token 消耗的曲线；计量性质另行标注 |
| `gantt.png` | 根据训练分数选定的方法的最佳策略所生成的代表性调度甘特图 |
| `report.md` | 区分 mock/真实、训练/测试/FT06、当前解/已知参考/求解器证明的实验报告 |

复现时使用同一 lock 文件、Python 版本、命令参数、种子和后端。mock 的候选、训练反馈与最终调度可按种子复现；墙钟时间和 CPU 时间会随机器负载变化，CP-SAT 在时间限制附近的求解状态也可能受环境影响。真实模型可能随服务版本或采样变化，即使提供 seed 也不保证按位一致。

先看 `report.md` 的后端标识和实际资源消耗，再看训练搜索曲线、留出集表现及 FT06。留出集只用于最终评估；不能据其结果反复调整参数后仍称为同一个未接触测试集。多个实验种子提供独立重复，但两个种子的演示不能支撑统计显著性或普遍优越性结论。

## 本次验证结果

本次实际执行了锁定依赖安装、全套 `pytest`（**59 passed**）、离线 `smoke`、两个搜索种子的默认 6×6 全方法实验、两个 CPU worker 的运行，以及从保存 JSON 重建报告。还在独立的干净虚拟环境中重新安装完整 lock 并跑通 smoke；两次 smoke 的策略、分数和调度一致。`pip check` 与 `git diff --check` 通过。

默认实验使用数据种子 2026、训练/留出各 8 个实例；以下均为 **mock 功能演示**。每个 Agent 方法每个搜索种子实际用了 12 次模拟生成和 15 次候选评估，随机搜索也评估 15 个候选；固定规则各评估 1 次。两个种子合计 72 次 **mock** 调用，没有真实模型调用。所有候选调度和最终调度合法率均为 100%。

| 方法 | seed 7 训练均值 | seed 7 留出均值 | seed 19 训练均值 | seed 19 留出均值 | FT06（7 / 19） |
| --- | --- | --- | --- | --- | --- |
| single | 120.500 | 120.000 | 128.125 | 123.250 | 60 / 69 |
| independent | 120.000 | 120.250 | 126.000 | 122.875 | 61 / 58 |
| cooperative | 120.000 | 120.250 | 126.000 | 122.875 | 61 / 58 |
| shortest_processing_time | 238.875 | 221.500 | 238.875 | 221.500 | 109 / 109 |
| most_work_remaining | 140.500 | 135.750 | 140.500 | 135.750 | 74 / 74 |
| random_search | 122.750 | 117.875 | 119.875 | 120.625 | 61 / 61 |

CP-SAT 在该次运行中为 16 个生成实例与 FT06 均返回 `OPTIMAL`；FT06 的目标与下界均为 55。搜索方法的 FT06 最好结果是 58，**没有匹配已知最优**。这两个 mock 种子下 independent 与 cooperative 相同，留出集随机搜索均值更低；这些数值不支持任何真实 LLM 协作优势结论。

可检查的演示原始数据、报告和图位于 [`examples/mock-demo/`](examples/mock-demo/)。演示元数据记录源码哈希及运行时工作区状态；时间计量按实际运行保留，不保证不同机器一致。**真实模型实验未运行**；兼容客户端已用本地合成响应测试其计量、失败与重定向行为。

## 实现位置

`src/swarm_scheduler_lab/` 包含实例生成、schema 策略、JobShopLib 派发、独立校验、CP-SAT 参考、预算账本、模型后端、三种搜索模式、CLI 和报告生成。`tests/` 覆盖约束违规、确定性、预算失败与并发、群间交换、训练/测试隔离、mock 复现及端到端 smoke。项目不需要创建 Git worktree，当前云任务已使用隔离 checkout。

## 限制与后续验证

第一版只搜索五个特征的线性加权派发规则与少量平局规则；没有任意程序发现、模型训练、在线逐动作决策或保证全局最优的搜索。小型随机实例与 FT06 不代表所有工业排程分布。默认短 CP-SAT 时间限制可能留下未证明最优的实例，必须保留上下界与状态。真实模型是否比随机搜索好、合作是否比独立搜索好，需要在冻结实验方案、足够重复和实际资源计量下验证。

## 来源与许可证

本项目使用库 API 编写独立实现，接口按实际安装的 JobShopLib 1.7.2 与 OR-Tools 9.12.4544 核对。参考的一手资料：

- [JobShopLib 源码与项目](https://github.com/Pabloo22/job_shop_lib)
- [JobShopLib Dispatching Rules 文档](https://job-shop-lib.readthedocs.io/en/stable/examples/03-Dispatching-Rules.html)（官方示例 notebook 源码也保存在该项目的 `docs/source/examples/03-Dispatching-Rules.ipynb`）
- [Google OR-Tools Job Shop 说明](https://developers.google.com/optimization/scheduling/job_shop)
- [OR-Tools 官方最小 Job Shop 示例](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/minimal_jobshop_sat.py)

本项目代码采用 [MIT License](LICENSE)。JobShopLib 的 MIT 声明、其打包 FT06 数据的来源，以及 OR-Tools 的 Apache-2.0 来源说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
