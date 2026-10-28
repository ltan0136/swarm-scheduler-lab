# MOCK 功能演示

这些结果来自固定种子的离线 mock，**不是真实 LLM swarm 实验**。
配置：数据种子 2026，训练/测试各 8 个 6×6 实例，搜索种子 7、19；每种 Agent 方法每个种子 12 次模拟生成、15 次候选评估。

- [实验报告](report.md)
- [完整原始 JSON](results.json)（紧凑格式以减少仓库体积）
- [统计 CSV](summary.csv)、[逐候选 CSV](evaluations.csv)
- [收敛曲线](convergence.png)、[代表性甘特图](gantt.png)

复现命令在仓库根 README。计时值保留实际观测；策略、分数及调度由种子确定。
源码哈希和运行时工作区状态记录在 JSON metadata 中。
当前演示不能证明协作优于其他搜索组织方式；真实模型实验未运行。
