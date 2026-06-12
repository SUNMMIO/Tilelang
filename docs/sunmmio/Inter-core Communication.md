# Sunmmio Inter-core Communication

## 1. Overview

Sunmmio adopts a 2D Mesh architecture, where different Cores hold their own local data and local calculation results. In this architecture, supporting only single-core local read and write is not enough; an inter-core data exchange mechanism must be provided to support the following scenarios:

- **Multi-core sharing of intermediate results**: For example, a core generates a result and broadcasts it to the same row, same column, or the entire Mesh.
- **Multi-core collaborative pipelining**: For example, one core completes a local calculation and sends the result directionally to another core for further processing.
- **Multi-core aggregation calculation**: For example, collecting local results from each core to perform all-gather or all-reduce.
- **High-level programming interface**: The underlying topology details should be shielded, allowing users to express communication intentions with higher-level semantics without having to manually write routing and offset calculation logic.

Therefore, the goal of this work is to establish a set of inter-core communication abstractions for Sunmmio Mesh in TileLang, mapping high-level semantics such as `broadcast`, `put`, `all_gather`, and `all_reduce` stably to underlying executable TIR communication primitives.

## 2. Design

The design is mainly divided into two layers: frontend interface design and backend operator design.

### 2.1 Python Frontend Interface Design

In `tilelang/language/comm.py`, a unified communication interface is designed with "ease of use" and "safety" as core considerations:

- **2D Mesh Topology**: Use `(row, col)` 2D coordinates to express core positions, which is more intuitive for Mesh topology.
- **Core ID Management**: Conversion between coordinates and linear Core IDs is handled in the frontend via `core_tuple_to_id` and `core_id_to_tuple`, reducing the burden of manual ID management.
- **Unified Direction**: A `direction` parameter is provided, supporting `horizontal`, `vertical`, and `all` directions, mapped to internal integer encodings.
- **Early Error Detection**: Checks for dtype, shape, buffer size, Mesh boundaries, and `size` parameters are performed in the frontend to catch errors during the DSL stage.

Core capabilities exposed at the interface layer include:

- `T.comm.broadcast(src, dst, src_core, direction="all", size=-1)`
- `T.comm.put(src, dst, src_core, dst_core, size=-1)`
- `T.comm.all_gather(send_buffer, recv_buffer, direction="all", size=-1)`
- `T.comm.all_reduce(buffer, out, reduce_type, direction, dim=-1, clear=True)`

This design allows users to write "what to communicate" rather than "how to communicate hop-by-hop."

### 2.2 C++ TileOp Abstraction Design

In `src/op/comm.h`, inter-core communication is modeled as a set of first-class TileLang operators:

- `BroadcastOpNode`
- `PutOpNode`
- `AllgatherOpNode`
- `AllreduceOpNode`

Each operator possesses two types of capabilities:

- `InferLayout`: Participates in layout inference and connects with preceding and succeeding operators.
- `Lower`: Converts high-level communication semantics into underlying `broadcast_` calls or combinations of communication and reduction logic.

This design ensures that communication is not just a collection of scattered intrinsic wrappers but is integrated into the TileOp system, naturally collaborating with existing compilation flows like `LayoutInference` and `LowerTileOp`.

### 2.3 Lowering Design

The focus of the backend design is to unify high-level communication semantics into a few underlying communication primitives to reduce complexity and improve maintainability.

- `broadcast` is directly lowered to the underlying `broadcast_`.
- `put` does not rely on a new hardware primitive but reuses `broadcast_`, simulating point-to-point transmission by selecting a direction and masking non-target cores.
- `all_gather` is implemented through a combination of multiple `broadcast_` calls, writing local data from each core into different segments of the receive buffer.
- `all_reduce` is implemented through a combination of "local reduce + all_gather + another reduce" for complex reduction semantics.

The overall design philosophy is to provide rich semantics at the frontend while reusing unified underlying primitives at the backend.

## 3. Implementation

### 3.1 Frontend Implementation

In `tilelang/language/comm.py`, the following specific implementations were completed:

- **Mesh Configuration**: Retrieve the target Mesh scale (`nrow`, `ncol`) via `get_sunmmio_device_mesh_config()`.
- **Core ID Management**: Implementation of `core_tuple_to_id` and `core_id_to_tuple`.
- **Parameter Validation**:
  - `broadcast` and `put` check dtype and shape compatibility of source/destination buffers.
  - Check if `src_core` and `dst_core` are within Mesh boundaries.
  - Check if `size` is out of bounds.
  - Check if `all_gather` receive buffer meets shape requirements after expansion.
  - Check if `all_reduce` parameters like `reduce_type`, `direction`, `dim`, and `clear` are valid.
- **TIR Intrinsic Encapsulation**: Finally generate `tl.tileop.comm_broadcast`, `tl.tileop.comm_put`, `tl.tileop.comm_allgather`, and `tl.tileop.comm_allreduce` calls via `tir.call_intrin`.

The significance of this step is that user intentions are converted into standardized TileOp calls recognizable by the compiler.

### 3.2 Lowering of Broadcast

In `src/op/comm.cc`, `BroadcastOpNode::Lower` handles the core implementation of broadcasting. Key points include:

- Ensure the current target is Sunmmio.
- Read Mesh row and column scales based on the target.
- Validate `src_core` boundaries and buffer sizes/offsets.
- For `horizontal` or `vertical` directions, generate a single `broadcast_`.
- For `direction == all`, decompose 2D broadcast into two stages:
  - **Stage 1**: Perform a vertical broadcast from the source core along the column.
  - **Stage 2**: Using the intermediate cores in each row as sources, perform horizontal broadcasts row by row.

This two-stage routing design fully utilizes Mesh structural features, enabling 2D broadcast through combinations of 1D primitives.

### 3.3 Lowering of Put

The core implementation idea of `PutOpNode::Lower` is to reuse `broadcast_` to simulate point-to-point communication:

- If source and destination are in the same row, perform a horizontal broadcast and use a mask to exclude all cores except the target.
- If source and destination are in the same column, perform a vertical broadcast and use a mask to exclude all cores except the target.
- If they are in different rows and columns, complete in two steps:
  - First, send data from the source core to an intermediate core along the column.
  - Then, the intermediate core sends it to the target core along the row.

This implementation avoids designing new hardware semantics for point-to-point sending, leveraging existing broadcast primitives instead.

### 3.4 Lowering of AllGather

`AllgatherOpNode::Lower` is handled in three categories based on direction:

- `horizontal`: Enumerate all source cores in each row, writing data sequentially into different offsets of the receive buffer.
- `vertical`: Enumerate all source cores in each column, writing data into the receive buffer.
- `all`: Executed in two steps:
  - First, perform in-row `all_gather`.
  - Then, perform column-wise broadcast on the in-row aggregation results.

An important implementation optimization: for `recv_buffer`, prioritize slicing along the first dimension rather than simple flattening. It only degrades to 1D flattened access if the layout is misaligned. This produces cleaner TIR and facilitates subsequent analysis and debugging.

### 3.5 Lowering of AllReduce

`AllreduceOpNode::Lower` is the most complex implementation. It is decomposed into composite operators:

- Perform a local `ReduceOp` first, reducing the input on the current core to `dst` or `dst_copy` along the specified dimension.
- If in-row or global reduction is needed, perform `AllgatherOp` on the local results, then perform another local `ReduceOp` on the gathered results.
- If in-column or global reduction is needed, similarly perform column-wise `AllgatherOp` and then another local `ReduceOp`.
- When `clear=False`, an additional `dst_copy` is maintained to ensure accumulation while preserving the original output content.

This approach reuses existing `ReduceOp` and `AllgatherOp`, building complex reduction logic on top of established components.

### 3.6 Testing and Verification

In `testing/python/language/test_tilelang_language_comm.py`, coverage testing was performed for communication interfaces and lowering results:

- `test_comm_python_api`: Verifies if the Python DSL correctly generates calls to `T.comm_broadcast`, `T.comm_put`, `T.comm_allgather`, etc.
- `test_comm_broadcast_lower`: Verifies if 2D broadcast is expanded into a vertical broadcast plus multiple horizontal broadcasts.
- `test_comm_put_lower`: Verifies if cross-row/cross-column point-to-point sending is expanded into two `broadcast_` calls.
- `test_comm_all_gather_lower`: Verifies if all-gather is expanded into a series of `broadcast_` calls with offsets.
- `test_comm_all_reduce_lower`: Provides expected TIR output samples for all-reduce lowering, demonstrating that the design covers the complete reduction path.

These tests use script-level comparison to directly inspect the structure and call sequence of lowered TIR, intuitively verifying the correct expansion of communication paths.

## 4. Result

The inter-core communication path has been fully established from DSL to underlying lowering. Key achievements include:

- Added Sunmmio Mesh multi-core communication capabilities in TileLang, supporting broadcast, point-to-point sending, all-gather, and all-reduce.
- Established a layered architecture from Python interface to TileOp and then to underlying implementations like `broadcast_` and `ReduceOp`, with clear interfaces and easy extensibility.
- Decomposed complex 2D communication and global reduction into combinations of 1D broadcasts and local reduces, significantly reducing implementation complexity.
- Added specialized tests for frontend APIs and lowering results, providing a reliable regression baseline for future iterations.

These results provide Sunmmio with a unified programming interface and a stable compilation path for multi-core data movement, laying the foundation for multi-core operators, distributed pipelining, and more complex parallel scheduling.
