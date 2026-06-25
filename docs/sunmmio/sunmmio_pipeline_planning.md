# Sunmmio Pipeline Planning 实现文档

## 1. 概述

`sunmmio_pipeline_planning.cc` 实现了面向分布式NPU的软件流水线调度框架。该文件是 `TileLang` 编译器中一个关键的优化 Pass，负责将高层循环体中的指令序列转换为高效的流水线调度方案。

### 1.1 核心目标

- 识别循环体中的**预取指令**（Prefetch Instructions），将访存操作提前执行
- 识别需要**多版本化**（Multiversioning）的缓冲区，消除WAW/WAR假依赖
- 通过**基于关键路径的列表调度**算法，在多异步引擎（ODMA、Tensor Core、Vector Core）间实现指令的高效重叠
- 生成**填充（Prologue）/ 内核（Body）/ 排空（Epilogue）** 三阶段调度序列

## 2. 核心数据结构

### 2.1 PipelineInstruction

表示一条指令的所有调度相关信息：

```cpp
class PipelineInstruction {
    int id;                  // 指令在单次迭代中的唯一ID
    int iter;                // 展开后的迭代索引
    string name;             // 格式: "{iter}-{id}"
    Stmt stmt;               // TVM IR 语句节点
    DeviceType device_type;  // ODMA / TensorCore / VectorCore
    bool is_prefetch;        // 是否为预取指令
    vector<BufferRegion> reads;   // 读取的缓冲区区域
    vector<BufferRegion> writes;  // 写入的缓冲区区域
    float scheduled_start;   // 调度后的绝对开始时间
    float scheduled_end;     // 调度后的绝对结束时间
    float delay;             // 执行延迟（时钟周期）
};
```

### 2.2 LocalDDG

单次迭代内的本地数据依赖图：

```cpp
struct LocalDDG {
    vector<LocalDependencyEdge> edges;           // 所有RAW依赖边
    vector<vector<int>> forward_predecessors;    // 前向依赖（D=0）
    vector<vector<int>> forward_successors;      // 前向后继
    vector<vector<int>> backward_predecessors;   // 后向依赖（D>0）
    vector<vector<int>> backward_successors;     // 后向后继
    unordered_map<BufferNode*, BufferAccessInfo> buffer_access_infos;
};
```

### 2.3 PipelineStageAssembly

展开后的三阶段指令窗口：

```cpp
struct PipelineStageAssembly {
    int iterations;                          // 内核迭代次数
    int epilogue_iterations;                 // 排空阶段迭代数
    vector<PipelineInstruction> prologue_instructions;  // 填充阶段
    vector<PipelineInstruction> body_instructions;      // 内核稳态
    vector<PipelineInstruction> epilogue_instructions;  // 排空阶段
};
```


## 3. 算法流程

### 3.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│  SunmmioPipelinePlanner::VisitStmt_(ForNode)                       │
│  入口：识别带有 "num_stages" 注解的循环                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段1：构建单次迭代的指令列表                                      │
│  - 提取 SeqStmt 中的每条指令                                       │
│  - 调用 HardwareMapper::Map 确定设备类型                          │
│  - 调用 CostModel::EstimateDelay 计算指令延迟                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段2.1：构建本地 DDG                                             │
│  LocalDDGBuilder::Build                                            │
│  - 分析每条指令的读/写缓冲区区域                                   │
│  - 建立 RAW 依赖边，标记距离 (D=0 或 D>0)                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段2.2：识别预取指令                                             │
│  PrefetchInstructionIdentifier::Identify                           │
│  - 种子条件：写全局内存、无前驱/后向依赖                          │
│  - 沿前向依赖图传播预取标记                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段2.3：识别多版本化缓冲区                                       │
│  MultiversioningIdentifier::Identify                               │
│  - 种子：预取指令的输出缓冲区                                     │
│  - 沿数据流传播，排除有循环携带依赖的缓冲区                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段3：展开指令窗口                                               │
│  PipelineWindowAssembler::Assemble                                 │
│  - 计算迭代数 = num_stages                                        │
│  - 构建 Prologue: 所有预取指令（iter=0）                          │
│  - 构建 Body: 计算指令(iter=0..S-1) + 预取指令(iter=1..S)         │
│  - 构建 Epilogue: 剩余计算指令                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段4：全局调度                                                   │
│  GlobalPipelineScheduler::Schedule                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 4.1 BuildDependencyGraph: 构建全局DAG                       │   │
│  │ 4.2 CalculateBottomLevels: 计算b-level优先级                │   │
│  │ 4.3 主调度循环: 事件驱动的列表调度                          │   │
│  │ 4.4 插入预取指令: 在空闲间隙中安插预取                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段5：持久化元数据                                               │
│  - 将调度结果写入循环注解 (prologue_orders, body_orders, ...)     │
│  - 记录 versioned_buffers 供后续内存分配 Pass 使用                │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 预取指令识别（PrefetchInstructionIdentifier）

**识别逻辑**：

1. **种子指令（Seed）** 需同时满足：
   - 是 DMA Copy / broadcast_ / layout_transform 操作
   - 有写操作
   - 无前向依赖（`forward_predecessors` 为空）
   - 无后向依赖（`backward_predecessors` 为空）
   - 所有读操作都来自全局内存（DRAM）

2. **传播规则**：沿 `forward_successors` 传播，要求：
   - 目标指令满足"有效预取指令"条件
   - 无后向依赖
   - 所有前驱都已被标记为预取

### 3.3 多版本化缓冲区识别（MultiversioningIdentifier）

**识别逻辑**：

1. **种子**：预取指令写入的非全局缓冲区
2. **传播条件**：
   - 该缓冲区无循环携带依赖（`first_read_index > first_write_index`）
   - 所有写入该缓冲区的指令，其所有输入缓冲区都已被标记为版本化

**目的**：标记出可以通过变量重命名消除WAW/WAR假依赖的缓冲区，为论文§4.2.2的实现提供依据。

### 3.4 窗口展开（PipelineWindowAssembler）

以 `num_stages = 3` 为例：

| 阶段 | 迭代范围 | 包含指令 |
|------|---------|---------|
| Prologue | iter=0 | 所有预取指令 |
| Body | iter=0..2 | 计算指令(0..2) + 预取指令(1..3) |
| Epilogue | 取决于循环余数 | 剩余的计算指令 |


### 3.5 全局调度（GlobalPipelineScheduler::Schedule）

#### 3.5.1 依赖图构建（BuildDependencyGraph）

按迭代顺序 + 指令ID顺序遍历所有指令，维护 `buffer_access_history`：

```
对于每条指令:
  1. 检查所有读操作 → 找最近的前驱写操作 → 添加依赖边
  2. 检查所有写操作 → 找最近的前驱写操作 → 添加依赖边
  3. 记录当前指令的读写操作到历史
```

**版本感知**：对于多版本化缓冲区，仅同一版本号的指令之间建立依赖。

#### 3.5.2 b-level 计算（CalculateBottomLevels）

```cpp
bottom_levels[i] = delay[i] + max(bottom_levels[succ])
```

计算顺序：逆拓扑序遍历所有指令。

#### 3.5.3 主调度循环

**优先级排序**（三级字典序）：
1. b-level 降序（关键路径优先）
2. iter 升序（迭代先后）
3. id 升序（指令原始顺序）

**事件驱动循环**：
```
while (主队列非空):
  1. 遍历主队列中所有未完成指令
  2. 若所有前驱已完成 → 贪心分配到空闲引擎
  3. 计算最快完成的事件时间 Δt
  4. 推进时间 t += Δt
  5. 更新各引擎状态（完成指令标记为 finished）
  6. 移除已完成的指令
```

#### 3.5.4 预取指令插入

在主调度完成后，将预取指令插入到空闲间隙中：

1. 构建预取子图，计算入度
2. 按迭代顺序 + ID顺序排序
3. 拓扑顺序调度每条预取指令
4. 在其依赖完成后的第一个足够大的空闲间隙中插入


## 4. 关键设计决策

### 4.1 预取指令的二阶段调度

这是实现中的一个重要工程决策：

1. **主调度**：只调度非预取指令（计算指令），以关键路径为导向
2. **预取插入**：在主调度完成后，将预取指令填入空闲间隙

**原因**：预取指令不应影响关键路径。若将它们与计算指令一起调度，可能抢占关键引擎时间，反而降低性能。

### 4.2 保守的访问重叠检测

`AccessOverlapChecker::Overlap` 仅使用 `buffer->same_as` 进行判断：

```cpp
static bool Overlap(const BufferRegion &lhs, const BufferRegion &rhs) {
    return lhs->buffer.same_as(rhs->buffer);
}
```

这是**保守近似**：
- ✅ 保证不会漏掉任何真实冲突
- ❌ 可能过度保守（同一buffer不同区域本不冲突）

设计文档中已预留扩展点（`Reserve an explicit extension point for future region-level overlap analysis`）。


## 5. 输出格式

### 5.1 循环注解

```tir
for (i, 0, N) {
  // annotations:
  //   num_stages: 3
  //   iterations: 3
  //   prologue_orders: ["0-0", "0-1"]      // 预取指令
  //   body_orders: ["0-2", "1-0", ...]     // 展开后的调度顺序
  //   epilogue_orders: ["2-1", ...]        // 排空阶段
  //   versioned_buffers: [buf_A, buf_B]    // 需多版本化的缓冲区
  //   used_buffers: [buf_A, buf_B, buf_C]  // 所有使用的缓冲区
  body
}
```

### 5.2 调试日志

开启 `debug=true` 时输出：
- `prologue.log` / `body.log` / `epilogue.log`：各阶段调度详情
- `body_graph.log`：依赖图节点和边信息


## 6. 用法

### 6.1 在 TVM Pass 管道中使用

```python
from tvm import tl

# 在编译流程中注册该Pass
with tvm.transform.PassContext(opt_level=3):
    mod = tl.transform.SunmmioPipelinePlanning(debug=True)(mod)
```

### 6.2 在算子中触发

```python
# TileLang 算子中通过注解触发
@T.prim_func
def my_kernel(...):
    for i in T.Pipelined(N, num_stages=4):
        # 流水线循环体
        T.copy(...)   # 会被识别为预取
        T.gemm(...)   # 计算指令
```

## 7. 扩展点

### 7.1 区域级重叠分析

将 `AccessOverlapChecker::Overlap` 从 buffer-level 升级为 region-level：

```cpp
static bool Overlap(const BufferRegion &lhs, const BufferRegion &rhs) {
    if (!lhs->buffer.same_as(rhs->buffer)) return false;
    // 添加实际的地址区间重叠检查
    return RegionOverlap(lhs->region, rhs->region);
}
```

### 7.2 Bank-Aware 调度

如论文改进讨论所述，可在 `GlobalPipelineScheduler` 中增加存储bank资源管理：

```cpp
class PipelineDevice {
    // 现有: 引擎资源 (ODMA/TC/VC)
    // 新增: 存储bank资源 (ASRAM_Ping/Pong, ...)
    std::vector<BankResource> banks_;
};
```

### 7.3 代价模型改进细化

`CostModel::EstimateDelay` 仍然可以继续改进以逼近硬件真实数据。
