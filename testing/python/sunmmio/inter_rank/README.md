# Inter-rank 测试

本目录的功能测试验证从 DSL 到最终 device TIR 的编译流程。
SUVM MLIR 生成和 A4E 编译约束由[点对点 codegen 测试](../codegen/test_dist_basic_opt_validate.py)
和[集合通信 codegen 测试](../codegen/test_dist_integration_opt_validate.py)验证。
正式测试按功能归档；新增回归用例放回对应功能文件，不再按开发阶段或审查批次新建文件。

## 功能分工

| 文件 | 覆盖内容 |
| --- | --- |
| `test_dist_foundation.py` | world_size、rank_id、MeshTensor rank placement 与元数据 |
| `test_dist_p2p.py` | 基础 put/wait、SRAM/DRAM payload、完整 lowering 与单卡边界 |
| `test_dist_signal_planning.py` | signal/SignalList 声明、kind/scope、容量分配与 sender 约束 |
| `test_dist_signal_state.py` | expected/generation 状态推进、多目的端、重复 wait 与循环复用 |
| `test_dist_state_storage.py` | 普通软件状态的 RSRAM/i32 存储、padding、布局传递及信号边界转换 |
| `test_dist_memory_state.py` | MEMORY uint32 软件存储、不检查累计上界及 i64 接口零扩展 |
| `test_dist_expected*.py` | 逻辑 u8 expected 最终峰值、分支合并和数值摘要的边界回归 |
| `test_dist_auxiliary_scope.py` | 通信辅助数组的 RSRAM 准入、显式 copy、非零 region 和标量 delta 边界 |
| `test_dist_region_validation.py` | dist 通用 region 检查接入、完整尾 tile、covered extent 与检查开关边界 |
| `test_dist_control_flow_admission.py` | SunMMIO 显式 return/break/continue 的前置临时准入及 scope/target 边界 |
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

多 rank kernel 在 DSL 检查 copy 区域、分配形状和 signal 数量时，以及在绑定 SunMMIO target 后的首次 TIR 化简前，使用同一套定宽规则。可证明保值的整数转换沿用原表达式。
标量窄化显式保留目标宽度的低位。8、16、32、64 位的同宽符号转换保留位模式，扩宽同时改变符号时先按源类型扩展。
需要回绕的动态向量窄化或无法证明为静态分配大小的回绕形状会在编译时诊断。
在 `T.Tiles` 内，局部值按实际标量或 tile 操作数转换，地址仍按标量索引处理。信号写入软件存储时复用相同规则，FLAGREG 的峰值检查使用原始 delta。
[整数转换 codegen 测试](../codegen/test_dist_integer_cast_policy_opt_validate.py)

普通 `put` 的本地 copy、跨 row 搬运和远端发送共用拓扑规划确认的源。嵌套 signed 64 位循环的次数证明使用可验证的外层逻辑 IV 范围。
消费式 `T.dist.wait_any` 的调用链须位于 kernel 或 `@T.macro` 中。需要按条件等待时，在显式 `if T.dist.has_pending(c)` 分支中取得成员，再将索引传给普通 Python helper。
[路由与惰性宏测试](../codegen/test_dist_semantics_opt_validate.py) · [循环边界测试](../codegen/test_dist_audit_kernel_opt_validate.py)

## 日常验证

使用 `my-tilelang` 环境，在仓库根目录按文件或用例运行，例如：

```bash
conda run -n my-tilelang python -m pytest -q testing/python/sunmmio/inter_rank/test_dist_completion.py
conda run -n my-tilelang python -m pytest -q testing/python/sunmmio/inter_rank/test_dist_submit.py -k early_return
```

前端参数检查尽量停在前端；单 pass 规则使用必要的前置 pass；每条主要 lowering 路径保留代表性的完整 device TIR 检查。
同一语义的参数变体优先参数化。不同层次的有效覆盖保留，公共规则的重复用例集中到负责该规则的文件。
优先检查 op 参数、Buffer 关联与控制流结构，避免依赖自动变量后缀或整段 TIR 文本。

日常按涉及的功能选择上表中的测试文件或用例。

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

生成的 TIR 日志保存在本地。
