# Work Summary Report: Sunmmio Shared Memory Merge & Rewrite (MergeSharedMemoryAllocationsSunmmio)

- Main implementation: [merge_shared_memory_allocations_sunmmio.cc]
- Tests: [test_tilelang_transform_sunmmio_merge_shared_memory_allocations.py]

---

## Overview

On Sunmmio, a single kernel may contain multiple shared memory (SMEM) allocations. This work delivers a TIR-level pass that merges multiple allocations within the same storage scope into a single allocation, reducing peak SMEM usage and allocation overhead via lifetime-aware reuse. The pass plans and rewrites allocations separately for Sunmmio-specific shared storage scopes (shared.asram/shared.wsram/shared.rsram), supports an optional “aggressive reuse” mode, and is compatible with the wait semantics required by asynchronous primitives such as DMA, MMA, and broadcast.

Key benefits:
- Merges multiple Allocate nodes in the same scope into one large buffer, reducing allocation count and peak footprint
- Reuses constant-sized buffers via linear-scan packing; appends dynamically-sized buffers sequentially
- Sunmmio-specific alignment: asram/wsram = 2048B, rsram = 64B
- Handles asynchronous pipelines: defers reclamation based on token/wait information to preserve correctness
- Enabled by default for the Sunmmio target, with detailed debug logs and unit-test coverage

---

## 1. Background and Goals

- Background: SMEM Allocate nodes appear at multiple sites in common pipelines; their lifetimes interleave and lead to wasted peak usage and extra allocation overhead.
- Goal: Merge multiple allocations into a single allocation without changing semantics, and reuse physical regions based on liveness; additionally satisfy stricter alignment and async-dependency requirements on Sunmmio.
- Scope: Process shared.asram/shared.wsram/shared.rsram independently; the pass only takes effect on the Sunmmio target.

---

## 2. Design

### 2.1 Scope Grouping and Placement
- Collect Allocate nodes per storage scope (asram/wsram/rsram), then plan and rewrite independently to avoid cross-scope interference.
- Insert a single merged Allocate at the entrance of thread_extent; rewrite all accesses within that thread scope.

### 2.2 Linearized Liveness Analysis
- Flatten the control-flow tree into a linear statement sequence and record the set of buffers touched by each statement.
- Compute the first and last touch for each buffer as its liveness boundaries.
- Support BufferLoad/Var references that appear at the same nesting level as Allocate, improving robustness for IR after Flatten/Lower.

### 2.3 Asynchronous-Semantics Compatibility
- Recognize sync_token_id attached to dma_copy/mma_sunmmio/broadcast_ calls, and treat the related buffers as constrained by wait requirements.
- When encountering wait_token, advance the “last touch” to an appropriate position; at the thread_extent exit, conservatively cover tokens that were not waited on within the scope.

### 2.4 Alignment Planning
- Mark stricter alignment requirements within DMA/MMA contexts, and take the maximum of device defaults and operator-specific constraints:
  - Sunmmio: asram/wsram = 2048B, rsram = 64B

### 2.5 Reuse Strategy
- Conservative: place kill points at the end sentinel of the Allocate’s scope, ensuring safe reuse.
- Aggressive (optional): when not inside loops, attribute touches to the tail of the current scope frame to tighten liveness and reduce peaks.

### 2.6 Arena Packing and Fallback
- Constant-sized buffers participate in linear-scan reuse (best-fit with a free list) with deterministic ordering.
- Dynamically-sized buffers are appended after the constant arena; total size is the maximum end offset, aligned per scope.
- If constant buffers are detected to overlap physically while their lifetimes overlap, fall back to sequential layout to guarantee correctness.

### 2.7 IR Rewriting
- Redirect all buffers in the target scope to the merged var, and add byte offsets (scaled appropriately by dtype) to indices.
- Rewrite tvm_access_ptr to use the merged var plus a unified offset.

---

## 3. Implementation

- Allocation collection and per-scope rewriting:
  - In [merge_shared_memory_allocations_sunmmio.cc], AllocateCollectorSunmmio groups allocations by scope; SharedMemoryRewriterSunmmio is constructed per scope to plan reuse and rewrite IR.
- Linearization and touch collection:
  - SharedMemLinearAccessPatternFinderSunmmio builds the linear statement sequence; it records touches from BufferStore/BufferLoad/Var references and uses begin/end sentinels for if/for/while/attr scopes.
- Async resource tracking:
  - During Call traversal, the pass extracts sync_token_id for dma/mma/broadcast_ and builds a token → buffers map; wait_token updates last-touch information; thread_extent exit performs conservative closure.
- Alignment and packing:
  - SharedMemoryAlignmentPlannerSunmmio annotates alignment requirements under DMA/MMA; LinearScanPack packs constant buffers; dynamic buffers are appended sequentially.
- Rewrite and placement:
  - Insert one merged Allocate at the first entry into thread_extent; rewrite Buffer accesses and tvm_access_ptr to merged var + offset.
- Pass wrapping and registration:
  - Name: tl.MergeSharedMemoryAllocationsSunmmio; enabled only on the Sunmmio target; controlled by ctx.config["tir.merge_static_smem"]; debug key: tl.debug_merge_shared_memory_allocations.

---

## 4. Testing and Validation

- Multi-scope merge with mixed sizes:
  - Case: test_merge_multi_scope_mixed_sizes (see test file)
  - Assert: exactly one Allocate remains for each scope: shared.asram/shared.wsram/shared.rsram.
- Aggressive vs. conservative comparison:
  - Case: test_aggressive_reuse_diff (enable_aggressive_merge vs. default strategy)
  - Assert: both strategies produce one Allocate; aggressive mode yields a smaller merged extent than conservative mode.
- Debuggability and observability:
  - With PassContext tl.debug_merge_shared_memory_allocations=True, logs include the merged buffer name, total size, per-buffer offsets and sizes, and liveness gen/kill events.

Test entry points and assertions are in: [test_tilelang_transform_merge_shared_memory_allocations_sunmmio.py](../../testing/python/transform/test_tilelang_transform_merge_shared_memory_allocations_sunmmio.py).

---

## 5. Usage and Configuration

- Pass: tl.MergeSharedMemoryAllocationsSunmmio(enable_aggressive_merge=False, asram_align_bytes=2048, wsram_align_bytes=2048, rsram_align_bytes=64)
- Target: takes effect only on the Sunmmio device target (TargetIsSunmmio)
- Default: enabled when ctx.config["tir.merge_static_smem"] = True
- Debug: ctx.config["tl.debug_merge_shared_memory_allocations"] = True enables detailed planning logs

---

## 6. Results

- Merge correctness: merges allocations per scope into a single allocation and rewrites accesses to merged buffer + offsets
- Peak reduction: constant-buffer reuse significantly reduces peak usage; dynamic buffers are appended with controlled total footprint
- Safety guarantees: token/wait-based handling prevents early reuse; conservative behavior inside loops; automatic fallback to sequential layout on detected conflicts
- Engineering readiness: enabled by default on Sunmmio, with detailed logs and unit-test regression coverage

---

## 7. Next Steps

- Improve interaction with preceding passes: refine touch attribution and kill-point lifting heuristics based on more real-world operator IR patterns
- Finer-grained alignment policies: tailor alignment per operator/mode using hardware access-width and bank models
- Dynamic-buffer reuse: explore “semi-static” packing strategies based on upper-bound estimates
- Joint regression with InjectSunmmioSync: run integrated validation on complex control flow and multi-core communication scenarios
