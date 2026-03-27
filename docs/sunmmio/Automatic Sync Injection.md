# Sunmmio Automatic Sync Injection (InjectSunmmioSync)

## 1. Overview

On Sunmmio, operations such as `dma_copy`, `mma_sunmmio`, and `broadcast_` exhibit distinct asynchronous characteristics. Asynchronous execution can improve parallelism and throughput, but it also brings a core challenge: if the compiler cannot accurately handle data dependencies, subsequent read/write operations are prone to timing errors. For example:

- A previous DMA has not yet completed, but a subsequent calculation starts reading from the destination buffer.
- A broadcast has not yet finished on all relevant cores, but subsequent calculations start consuming the broadcast result.
- Dependencies exist across iterations or branches in loops and conditional statements; incomplete synchronization insertion may lead to deadlocks, dirty reads, or write-after-write conflicts.

Therefore, the goal of automatic synchronization injection is to have the compiler automatically analyze asynchronous dependencies in TIR and insert appropriate synchronization primitives such as `wait_token`, `barrier_init`, and `barrier_arrive_and_wait`, ensuring program semantic correctness without increasing the user's manual burden.

## 2. Design

The core design of synchronization injection is "using tokens to manage asynchronous completion status and barriers to manage cross-core coordination status."

### 2.1 Synchronization Primitive Design

Synchronization-related intrinsics are defined and registered in `src/op/builtin.h` and `src/op/builtin.cc`, including:

- `sync_token_id(token_id)`: Tags an asynchronous operation with a token, representing its completion identifier.
- `wait_token(token_id)`: Waits for the asynchronous operation corresponding to a specific token to complete.
- `sync_null_token(token_id)`: Declares a pre-completed null token, used to break dependency initialization issues in loop scenarios.
- `barrier_init(barrier_id, read_core, write_cores...)`: Initializes a cross-core synchronization barrier.
- `barrier_arrive_and_wait(barrier_id)`: Synchronously waits on a barrier.

Key points:
- Tokens are better suited for expressing the completion relationship of individual asynchronous operations.
- Barriers are better suited for expressing synchronization points involving multiple cores, such as broadcasts.

The combination of both covers both single-core asynchronous pipelining and multi-core collaborative communication.

### 2.2 Pass Structure Design

`InjectSunmmioSync` is not a simple linear instrumenter but is composed of multiple collaborating Rewriters:

1. **InjectSyncRewriter**: Responsible for identifying asynchronous operations, allocating tokens, performing RAW/WAR/WAW dependency analysis, and initially inserting waits.
2. **BarrierExtractRewriter**: Responsible for organizing broadcast-related barriers so they can be correctly extracted and reused within the control flow.
3. **DeviceScopeWaitRewriter**: Responsible for adding final waits for uncompleted tokens and barriers at the end of the device execution scope.
4. **EliminateRedundancyRewriter**: Responsible for deleting redundant synchronizations after ensuring correctness, reducing unnecessary waiting overhead.

This multi-stage design ensures that the pass possesses both "correctness priority" and "performance optimization" capabilities.

### 2.3 Dependency Analysis Design

Key designs in the dependency analysis layer include:

- Treating `dma_copy`, `mma_sunmmio`, and `broadcast_` uniformly as asynchronous resources.
- Collecting read/write information for buffer regions from each statement.
- Identifying RAW, WAR, and WAW dependencies based on region intersection.
- Maintaining the mapping relationship between tokens and barriers specifically for broadcasts, as broadcast completion depends not only on the local token but also on whether all relevant cores have reached the synchronization point.
- Propagating the "set of resources to be completed" through control flows like `if`, `for`, and `while`, preventing incomplete global semantics caused by only local synchronization.

By design, the pass does not simply insert waits in textual order but explicitly maintains a set of states representing "unresolved tokens and unreached barriers."

## 3. Implementation

### 3.1 Asynchronous Operation Identification and Token Allocation

In `src/transform/inject_sunmmio_sync.cc`, the pass first identifies asynchronous operations requiring synchronization management:

- `dma_copy`
- `mma_sunmmio`
- `broadcast_`

For each asynchronous operation encountered, a unique token is allocated, and `sync_token_id(token_id)` is appended to the corresponding call. Subsequently, the compiler checks if subsequent read/write operations conflict with the buffer regions involved in that operation. If a conflict is detected, `wait_token(token_id)` is inserted at the appropriate position.

This approach transforms "when an asynchronous operation completes" from an implicit assumption into an explicitly traceable state, providing a foundation for subsequent control flow analysis.

### 3.2 Barrier Establishment and Broadcast Synchronization

Broadcast is the most unique type of operation in synchronization injection because it naturally involves multiple cores. A barrier mechanism is added to the implementation for this:

- When `broadcast_` is encountered, both a token and a barrier are allocated.
- Based on the broadcast direction and the source core's position, the pass deduces which cores are write participants and which are read participants.
- `barrier_init` is inserted to explicitly establish a communication synchronization point.
- `barrier_arrive_and_wait` is inserted at subsequent appropriate positions to ensure all cores involved in the broadcast reach a consistent state before proceeding.

Thus, the "completion" of a broadcast is no longer just the completion of a single token but a multi-core collaborative completion.

### 3.3 Control Flow and Loop Handling

One of the most challenging parts of synchronization injection is handling `if/else` and loops.

In **conditional branches**:
- The compiler separately analyzes token and barrier states in then/else branches.
- Synchronization relationships that need to remain effective outside the branches are extracted.
- This prevents inconsistent states after merging branches where one branch might have waited while another did not.

In **loops**:
- The compiler analyzes loop-carried dependencies—whether an asynchronous operation from a previous iteration affects the next iteration.
- `sync_null_token` is used for initialization for tokens that need to be preset before the loop starts.
- For broadcast-related loops, stable barrier initialization and rotation synchronization logic are maintained to prevent deadlocks or waiting order errors.

This allows synchronization injection to cover complex control flows common in real kernels, beyond simple linear code.

### 3.4 Device Scope Exit Guarantee

`DeviceScopeWaitRewriter` handles "cleanup synchronization" before the device execution scope ends:

- Collecting all tokens that have not been explicitly waited for.
- Collecting all barriers that have not yet completed.
- Uniformly adding `wait_token` and `barrier_arrive_and_wait` before the kernel exits.

This design is crucial because the final operations in an asynchronous pipeline often lack subsequent consumer statements to trigger a wait. Without device scope cleanup logic, the kernel might terminate prematurely before underlying write-backs are complete, leading to incomplete results or runtime errors.

### 3.5 Redundant Synchronization Elimination

Relying solely on initial dependency injection often results in a "conservative but heavy" synchronization result. To reduce overhead, a redundancy elimination phase is included:

- Using `AsyncResourceCollector` and `PendingAnalyzer` to collect tokens/barriers that remain unresolved under the current control flow.
- If a token has already been waited for previously, redundant `wait_token` calls are removed.
- If a barrier has already been satisfied along the path, subsequent redundant waits are eliminated.
- States are precisely propagated through if/loop structures to avoid errors caused by simple local deletions.

This step ensures the pass is not only correct but also performance-conscious.

### 3.6 Testing and Verification

In `testing/python/transform/test_tilelang_transform_inject_sunmmio_sync.py`, synchronization injection is verified across different scenarios:

- `test_inject_sunmmio_sync_dma`: Verifies correct token and wait ordering between DMAs.
- `test_inject_sunmmio_sync_mma`: Verifies the dependency chain between DMAs and MMA before and after GEMM.
- `test_inject_sunmmio_sync_broadcast`: Verifies correct insertion of `barrier_init` and `barrier_arrive_and_wait` in broadcast scenarios.
- `test_inject_sunmmio_sync_if`: Verifies that IR after barrier/token extraction in conditional branches matches expectations.
- `test_inject_sunmmio_sync_loop`: Verifies `sync_null_token` in loops, synchronization rotation within the loop body, and final cleanup waits at the device end.

These tests go beyond checking for API names; they verify the sequence of synchronization calls. Some scenarios use precise IR fragment comparisons, imposing strong constraints on pass correctness.

## 4. Result

The automatic synchronization injection has been fully implemented, from defining underlying primitives to automatic compiler insertion and specialized testing. Key benefits include:

- The compiler automatically adds synchronization for asynchronous operations like DMA, MMA, and Broadcast, eliminating reliance on manually written wait logic.
- Support for dual mechanisms (tokens and barriers) to cover both individual asynchronous operations and multi-core broadcast coordination.
- Correct handling of complex scenarios including linear code, conditional branches, loops, and device scope exit, significantly improving the engineering usability of the pass.
- Redundant synchronizations are further eliminated on top of guaranteed correctness, reducing unnecessary waiting overhead.
- A comprehensive regression verification system has been established through specialized tests, providing a stable foundation for further expanding asynchronous operators and communication patterns.

In conclusion, the implementation of `InjectSunmmioSync` transforms asynchronous execution on Sunmmio from "expressible" to "safely executable." Together with inter-core communication capabilities, it forms a critical loop in Sunmmio multi-core compilation support: enabling cross-core data movement while ensuring it remains correct in complex asynchronous environments.
