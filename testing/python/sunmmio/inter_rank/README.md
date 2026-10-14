# Inter-rank 测试

本目录验证 DSL 到最终 device TIR，不执行 codegen、设备运行或数值正确性测试。
正式测试按功能归档；新增回归用例放回对应功能文件，不再按开发阶段或审查批次新建文件。

## 功能分工

| 文件 | 覆盖内容 |
| --- | --- |
| `test_dist_foundation.py` | world_size、rank_id、MeshTensor rank placement 与元数据 |
| `test_dist_p2p.py` | 基础 put/wait、SRAM/DRAM payload、完整 lowering 与单卡边界 |
| `test_dist_signal_planning.py` | signal/SignalList 声明、kind/scope、容量分配与 sender 约束 |
| `test_dist_signal_state.py` | expected/generation 状态推进、多目的端、重复 wait 与循环复用 |
| `test_dist_expect_control_flow.py` | 分支和 For/While 中的 expected 推导、auto/manual 合法性与通信阶段顺序 |
| `test_dist_manual_expect.py` | 用户指定 expected 增量、动态 member 索引和协议 manual 接入 |
| `test_dist_completion.py` | Completion 快照、pending、wait_any/wait_all 与复用限制 |
| `test_dist_row_routing.py` | 对端/跨 row/同 Rank 路由、src_rank、offset 和 staging |
| `test_dist_routing_memory_signal.py` | MEMORY signal 与跨 row 路由的组合、自动回退与约束 |
| `test_dist_collective_all*.py` | 各 collective 的前端、展开、协议接入与代表性 device TIR |
| `test_dist_collective_signal_ownership.py` | collective 聚合 signal 的独占、混用和复用边界 |
| `test_dist_submit.py` | 发送队列、submit、sender wait 与当前退出收尾行为 |
| `test_dist_barrier.py` | put_signal、barrier_arrive 和 barrier 语法糖 |
| `test_dist_pipeline_admission.py` | dist op 与自动 pipeline 的准入边界 |
| `test_lowering.py` | lowering 工具的 pass 快照与 device kernel 提取 |

`lowering.py` 提供真实编译流水线和 pass 快照；`tir_test_utils.py` 提供公共 TIR 查询与单阶段 lowering。
kernel factory 留在所属功能文件中，测试文件之间不相互导入。

## 日常验证

使用 `my-tilelang` 环境，在仓库根目录按文件或用例运行，例如：

```bash
conda run -n my-tilelang python -m pytest -q testing/python/sunmmio/inter_rank/test_dist_completion.py
conda run -n my-tilelang python -m pytest -q testing/python/sunmmio/inter_rank/test_dist_submit.py -k early_return
```

前端参数检查尽量停在前端；单 pass 规则使用必要的前置 pass；每条主要 lowering 路径保留代表性的完整 device TIR 检查。
同一语义的参数变体优先参数化。不同层次的有效覆盖保留，公共规则的重复用例集中到负责该规则的文件。
优先检查 op 参数、Buffer 关联与控制流结构，避免依赖自动变量后缀或整段 TIR 文本。

日常只运行相关用例；完整回归另行指定。不要直接递归运行本目录：`kernel/` 的迁移和检查单独进行。
本地的 `kernel/` 应用示例及 `kernel/history/` 草稿独立维护，不属于这里的正式回归集合。
测试不读取本说明或 `doc/` 中的文档。

## 显式打印 TIR

原 `test_dist_pass_tir_print.py` 已改为 `show_dist_tir.py`，保留全部 20 个正向展示和关键断言，不再被普通 pytest 收集。

```bash
# 列出示例，不编译
conda run -n my-tilelang python -m testing.python.sunmmio.inter_rank.show_dist_tir --list

# 默认通过 mod.show() 显示；不指定 --case 时运行全部示例
conda run -n my-tilelang python -m testing.python.sunmmio.inter_rank.show_dist_tir --case collective_all_gather

# 指定写入日志
conda run -n my-tilelang python -m testing.python.sunmmio.inter_rank.show_dist_tir --case cross_row_routing --output log
```

日志仍写入本目录的 `log/<kernel 名>.log`，不纳入提交。
