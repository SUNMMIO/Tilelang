## **Function description**

### **Purpose**

该函数为tilelang/engine/phase.py中**LowerAndLegalize**函数下的一个pass，目的为Infer memory layouts for fragments and shared memory。

### **Semantics**

函数在tilelang/engine/phase.py中被调用：

```
mod = tilelang.transform.LayoutInference()(mod)
```

逐层跳转到src/transform/layout_inference.cc中：

```
tvm::transform::Pass LayoutInference() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    f.CopyOnWrite()->body = ParallelLoopTransformer::Substitute(f->body);
    ThreadBindingCollector collector;
    collector(f->body);
    bool has_thread_binding = !collector.thread_binding_.empty();
    bool skip_thread_partition = !has_thread_binding;
    return LayoutInferencer::Substitute(std::move(f), skip_thread_partition);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LayoutInference", {});
}
```

```
  static PrimFunc Substitute(PrimFunc f, bool skip_thread_partition = false) {
    arith::Analyzer analyzer;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = ParallelLoopFuser::Fuse(f->body);
    BufferUseDefCollector collector(skip_thread_partition);
    collector.Collect(f);
    auto result = collector.Run();
    LayoutInferencer substituter(result, skip_thread_partition, &analyzer);
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }
```

核心部分在collector.Run()中：

```
// step 1: infer strict layout
for (int i = 0; i < num_infer; i++) {
  RunInferStep(i, InferLevel::kStrict, false, layout_map, strict_layout_map,
               q, in_queue);
}

// step 2: infer common layout with BFS
FinishInferQueue(InferLevel::kCommon, layout_map, strict_layout_map, q,
                 in_queue);
                 
// step 3: relax constraints to free and re-run
InferInFreeMode(layout_map, strict_layout_map);

// step 4: finalize alias layouts by Var
// For each storage var, if any buffer in the group has a layout,
// propagate (reshape if needed) to the rest to ensure completeness.
```

这三者都是对以下函数的调用：

```
// Run InferLayout
auto updates =
    next->InferLayout(LayoutInferArgs{target_, thread_bounds, layout_map,
                                      cur_analyzer, buffer_oob},
                      level);
```

## **Implementation details**

### **Structure**

src/op/copy.cc中根据不同情况将copy分为以下几个类别并分类处理：

```
/// Copy instruction types for different memory access patterns
enum class CopyInst : uint8_t {
  kNormal = 0,    // utilize ldg/stg or cpasync or any buffer copy
  kLDSM = 1,      // ldmatrix memory copy
  kSTSM = 2,      // stmatrix memory copy
  kBulkLoad = 3,  // utilize tma load
  kBulkStore = 4, // utilize tma store
  // we should separate the bulk load and store for 1d and multi-dim
  // as they have different memory access patterns
  kBulkLoad1D = 5,  // utilize tma load 1d
  kBulkStore1D = 6, // utilize tma store 1d
  kTMemLoad = 7,    // tcgen05.ld (tensor memory -> register)
  kTMemStore = 8,   // tcgen05.st (register -> tensor memory)

  // dma
  kDMALoad = 9,
  kDMAStore = 10,
};
```

DMA的copy依然属于copy的一种，因此这里将DMA的相关代码添加到了copy.cc中。

### **Implementation**

首先需要鉴别出是哪种copy：

```
auto copy_inst = GetCopyInst(target, disable_tma_lower || disable_tma,
                           T.layout_map, T.analyzer, T.buffer_oob);
```

仿照TMA的例子，添加了以下两个函数：

```
/*!
* \brief Check if dma load is supported.
*/
bool CheckDMALoad(Target target, arith::Analyzer *analyzer,
                 bool check_last_dim = true) const;

/*!
* \brief Check if dma store is supported.
*/
bool CheckDMAStore(Target target, arith::Analyzer *analyzer,
                 bool check_last_dim = true) const;

// 1. arch must support zpu
// 2. src and dst must be global and shared
// 3. check shape.
// last dim of src * dtype.bits() must be a multiple of 16
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
// now we check src (gmem) as tma box dim is deduced from src
// 4. src and dst must have the same dtype
```

确定需要DMA copy操作后，对node推断layout：

```
  if (copy_inst == CopyInst::kDMALoad || copy_inst == CopyInst::kDMAStore) {
    // if can apply swizzling, we skip layout inference
    // for dma load/store, we can directly apply the layout of normal copy
    // This must be a global/shared layout, so we can skip the parallel op
    // layout inference (parallel layout inference only annotate the loop layout
    // and the register layout).
    // the same implementation as TMA
    bool is_load = copy_inst == CopyInst::kDMALoad;
    Buffer global_tensor = is_load ? src : dst;
    Buffer shared_tensor = is_load ? dst : src;
    // check shared layout is non-swizzle
    // skip layout inference if shared layout is already annotated
    if (level == InferLevel::kFree && !T.layout_map.count(shared_tensor)) {
      // create a new layout map for tma linear layout
      Layout linear_layout = ComputeLinearLayout(shared_tensor);
      return Map<Buffer, Layout>({{shared_tensor, linear_layout}});
    }
    return {};
  }
```

```
 * @brief Compute a linearized shared-memory layout used for TMA transfers.
 *
 * Creates a Layout that maps an N-D shared tensor into a 1-D-like ordering
 * suitable for TMA by blocking each dimension into 256-element tiles and
 * splitting each original index into a quotient and remainder. Effectively
 * transforms each index i_k into two coordinates: floor(i_k / 256) and
 * i_k % 256, producing an ordering equivalent to concatenating all quotients
 * followed by all remainders.
 *
// [i, j] -> [i // 256, j // 256, i % 256, j % 256]
```

### **Unit test**

目前该部分没有真正的功能，只是添加了对于zpu的接口，实际作用与TMA一致，因此只需要验证TMA的infer结果与DMA一致即可。

手动修改copy_inst以改变程序的条件：

```
if (copy_inst == CopyInst::kBulkLoad)
	copy_inst = CopyInst::kDMALoad;
```

最终整理结果如下：

```
DMA

kNormal = 0,    // utilize ldg/stg or cpasync or any buffer copy
kDMALoad = 9,

[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 9
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 9
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 0
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 9
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 9
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 0
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 9
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 9
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 0
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 0
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 0
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 0
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 0
[14:46:05] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:490: 0
C_local inferenced layout:
  Shape: [32, 32] -> [8]
  Thread: _j // 16 * 64 + _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2
  Index:  [_j % 16 // 8 * 4 + _i % 16 // 8 * 2 + _j % 2]

Normal

kNormal = 0,    // utilize ldg/stg or cpasync or any buffer copy
kBulkLoad = 3,  // utilize tma load

[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 3
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 3
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 0
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 3
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 3
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 0
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 3
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 3
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 0
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 0
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 0
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 0
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 0
[14:41:34] /home/wanghaoze/Tilelang-Mesh/src/op/copy.cc:488: 0
C_local inferenced layout:
  Shape: [32, 32] -> [8]
  Thread: _j // 16 * 64 + _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2
  Index:  [_j % 16 // 8 * 4 + _i % 16 // 8 * 2 + _j % 2]
```

两次输出结果完全一致。
