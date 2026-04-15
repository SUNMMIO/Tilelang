# Tile-Level Fill and Reduce: Adapting TileLang Operators to a Vector-Core Execution Model
## Tile Ops
### Motivation
Thanks to the underlying structure of Serial + Vectorized loops, many simple element-wise operators are naturally supported. These include `+, -, *, /, exp, exp2, log, sin, cos, sqrt, floor, ceil, trunc, round, abs ...` and so on. However, for operators involving memory initialization and reduction, we still need specialized implementations, primarily `T.fill()`, `T.clear()`, and `T.reduce()`.

### Implementation
The main modifications are in the `LowerTileOp` Pass and the lowering part of related Ops (`fill()`, `reduce()`). Since `T.clear()` is a special case of `T.fill()`, implementing the latter essentially resolves the former.

#### **Operator Interface Extension**:
First, to allow Tile-related operators to access Tile partition information during the Lowering phase, we enhanced the general interface for operator lowering:

- Extended `LowerArgs` by adding a `TileViewMap` field to pass `TileView` information during operator lowering.
- Updated the `LowerTileOp` pass: Captured `TileView` from the `annotate_tileview` annotation of the buffer, and propagated it to each operator during the lowering process. This ensures that every operator can generate an appropriate underlying loop structure based on Tile semantics.

#### T.fill() and T.clear() Operators
The basic logic is to use a tile loop to represent the tile iteration, and then use a buffer store to "assign" values within the loop body. This mainly involves two scenarios: filling the entire buffer, and filling a buffer region (a specific area of the buffer).
The lowering of fill/clear is mainly implemented in the `fill.cc` file. The key points are as follows:
1. **Scope Enforcement**: Added strict checks to ensure that the target buffer for fill/clear under Sunmmio is located in `shared.rsram`, meeting the hardware requirements of the vector core for data placement.
2. **Tiled Loop Generation**: Based on the `TileView` corresponding to the target buffer, `T.fill()` is lowered into a serial loop nest iterating over tiles. This prepares for the subsequent `TilesLoop` pass to insert the internal `serial + vectorized` loops within the tile.
3. **Critical Annotation Injection**: Automatically attaches critical annotations required for `TilesLoop` expansion (e.g., `tile.buffer_new_shape`, `tile.loop_stage`, `tile.execution`, etc.), enabling subsequent passes to recognize and expand them into a vectorized instruction-level structure.
4. **Region Support**: Added support for filling a partial region of the buffer (e.g., `A_shared[..., 0:128, ...]`). Strict alignment checks are performed on the offset and extent of the region, requiring them to be divisible by the tile size, thereby ensuring the region can be accurately mapped to the tile-level loops.
5. **Clear Reusing Fill Logic**: `T.clear()` essentially reuses the lowering logic of fill, only fixing the fill value to 0 (while maintaining the identical tile-level loop and annotation structure).

#### Result

```python
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), T.ceildiv(B, block_B), threads=128) as (bx, by, bz):
        A_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
        T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})

        # 1. Fill the entire buffer
        T.fill(A_shared, T.float16(1.0))

        # 2. Fill a region of the buffer
        T.fill(A_shared[0:block_B, 0:block_M//2, 0:block_N], T.float16(2.0))

        # 3. Fill a region with offset
        T.fill(A_shared[0:block_B, block_M//2:block_M, 0:block_N], T.float16(3.0))
        # 4. T.clear()
        T.clear(A_shared)
```

After the `LowerTileOp` pass (Note: only the lowered result of `T.clear(A_shared)` is shown, same below):

```python
    ...
    for i in T.serial(16, annotations={"tile.buffer_new_shape": [16, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.loop_parallel": 1, "tile.loop_stage": 1, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
        for j in T.serial(8, annotations={"tile.buffer_new_shape": [16, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.execution": 1, "tile.loop_parallel": 1, "tile.loop_stage": 1, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
            for k in T.serial(4, annotations={"tile.buffer_new_shape": [16, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.execution": 1, "tile.loop_parallel": 1, "tile.loop_stage": 1, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
                # Express T.clear using buffer store (i.e., T.fill(A_shared, 0))
                A_shared[i, j, k] = T.Cast("float16", 0)
    ...
```

After the `TilesLoop` pass:

```python
    for i in T.serial(16, annotations={"tile.buffer_new_shape": [16, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.loop_parallel": 1, "tile.loop_stage": 2, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
        for j in T.serial(8, annotations={"tile.buffer_new_shape": [16, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.execution": 1, "tile.loop_parallel": 1, "tile.loop_stage": 2, "tile.scope_entry": 1, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
            for k in T.serial(4, annotations={"tile.buffer_new_shape": [16, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.execution": 1, "tile.loop_parallel": 1, "tile.loop_stage": 2, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
                # Similar to T.Tiles(), inserts two layers of loops; replaces the buffer indices
                for ki in T.serial(32, annotations={"tile.interior": 1, "tile.interior_axis": 0, "tile.loop_stage": 2, "tile.tiled_buffer": A_shared.data}):
                    for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1, "tile.loop_stage": 2, "tile.tiled_buffer": A_shared.data}):
                        # Express T.clear using buffer store
                        A_shared[i, j * 32 + ki, k * 32 + kj] = T.Cast("float16", 0)
```

#### T.reduce()

The case for `reduce` is slightly more complex. In general, the operator implementation differs slightly depending on the reduction axis. When the reduction axis belongs to a tile partition axis, it requires accumulation between tiles first, followed by an in-tile reduction on the accumulated result; in this case, we need to register an intrinsic for in-tile reduction. When the reduction axis is not a tile partition axis, only accumulation between tiles is needed.
The lowering of reduce is mainly implemented in `reduce.cc`. The key points of the entire operator implementation are as follows:

1. **Scope Enforcement**: Added strict checks to ensure that both the input and output of reduce under Sunmmio are located in `shared.rsram` (or allowed by frontend-specific paths like rsram/asram/wsram), to meet the vector core's hardware requirements.
2. **Loop Reordering & Accumulator Reuse**: Constructed a multi-level loop nest and explicitly placed the reduce axis at the innermost level (innermost spatial loop), enabling efficient reuse of the accumulator `acc` buffer across tile iterations and reducing repeated allocation/write-back of intermediate data.
3. **Unified Guarded Reduction Logic**: Adopted a unified loop body and controlled behavior at different stages using guards:
   - `if rv == 0`: Initialize the accumulator (clear to zero).
   - `if rv == last`: Execute the final reduction/write-back (finalize/write-back).
   This design allows spatial dimensions and reduction dimensions to correctly manage state under Tile semantics (especially the initialization and finalization needed for in-tile reduction).
4. **Hardware Primitive Mapping**: When reduction occurs along a tile axis, it integrates `vector_core_in_tile_reduce`. The implementation maps high-level operators (like `abssum`) to hardware-supported reduction primitives (like `sum`), and performs element-wise preprocessing during the accumulation stage (like applying `fabs` to the input), ensuring semantic consistency.
5. **Memory Optimization via BufferRegion**: Uses `BufferRegion` to pass the target output area directly into the in-tile reduction builtin, avoiding the introduction of additional intermediate output buffers (no longer needing a temporary buffer to save the in-tile reduce result), reducing memory overhead and simplifying the IR.
6. **Annotation Alignment**: Injects critical tile-level annotations (e.g., `tile.loop_stage=kTiled`, `tile.execution`, `tile.scope_entry`), ensuring that the IR structure generated by reduce is consistent with the tiles loop system of `T.Tiles()`/`T.fill()`, allowing seamless integration with the subsequent `TilesLoop` and Codegen pipeline.
7. Finally, there is the **Frontend Macro Logic Optimization** (`reduce_op.py`):
   - **Sunmmio Specialized Path**: In `reduce_macro`, detects the Sunmmio target or specific memory scopes (`shared.rsram/asram/wsram`) to bypass traditional GPU paths (like fragment allocation) and directly go through hardware-friendly lowering.
   - **Relaxed Shape Validation**: Implemented a more flexible shape checking mechanism, ignoring unit dimensions with size=1, to accommodate hardware-driven alignment/padding requirements while maintaining logical correctness.

### Result

Taking a tensor with shape `(b, m, n)` as an example, there are two cases for the reduction axis: tile axis (`m` or `n`), and non-tile axis (`b`). Below is the user's syntax:

```python
    with T.Kernel(T.ceildiv(N, block_N) if reduce_axis != 2 else T.ceildiv(M, block_M),
                    T.ceildiv(B, block_B) if reduce_axis != 0 else T.ceildiv(M, block_M), threads=128) as (bx, bz):
        A_shared = T.alloc_shared((block_B, block_M, block_N), dtype, scope="shared.rsram")
        Out_shared = T.alloc_shared(out_shape_block, dtype, scope="shared.rsram")

        T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})
        # Different reduction axes
        if reduce_axis == 2: # Reduce N
            T.annotate_tileview({Out_shared: make_tileview(Out_shared, [tile_size[0]], [-1])})
        elif reduce_axis == 1: # Reduce M
            T.annotate_tileview({Out_shared: make_tileview(Out_shared, [tile_size[1]], [-1])})
        else: # Reduce B
            T.annotate_tileview({Out_shared: make_tileview(Out_shared, tile_size, index_map)})

        if reduce_axis != 0:
            T.copy(A[bz * block_B:(bz + 1) * block_B, 0:block_M, 0:block_N], A_shared)
        else:
            T.copy(A[0:block_B, 0:block_M, 0:block_N], A_shared)
        # Execute reduce operation
        T.reduce_abssum(A_shared, Out_shared, dim=reduce_axis, clear=False)
        ...
```

After the `LowerTileOp` pass (below is the IR generated when reducing along the `m` axis):

```python


    with T.block("reduce_tile_op", no_realize=True):
        T.reads()
        T.writes()
        # Allocate intermediate buffer acc
        Out_shared_acc = T.handle("float16", "shared.rsram")
        T.block_attr({"tileview_map": {Out_shared_acc: metadata["tl.TileView"][2]}})
        Out_shared_acc_1 = T.alloc_buffer((1, 32, 32), "float16", data=Out_shared_acc, scope="shared.rsram")
        # Allocate intermediate buffer res (for in-tile reduce result)
        Out_shared_res = T.alloc_buffer((1, 32), "float16", scope="shared.rsram")
        for i in T.serial(32, annotations={"tile.buffer_new_shape": [32, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.loop_parallel": 1, "tile.loop_stage": 2, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
            for k in T.serial(4, annotations={"tile.buffer_new_shape": [32, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.execution": 1, "tile.loop_parallel": 1, "tile.loop_stage": 2, "tile.scope_entry": 1, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
                # The innermost loop is j, corresponding to the m axis
                for j in T.serial(8, annotations={"tile.buffer_new_shape": [32, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.execution": 1, "tile.loop_parallel": 0, "tile.loop_stage": 2, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
                    # Initialize intermediate acc buffer
                    if j == 0:
                        for ki in T.serial(32, annotations={"tile.interior": 1, "tile.interior_axis": 0, "tile.loop_stage": 2, "tile.tiled_buffer": A_shared.data}):
                            for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1, "tile.loop_stage": 2, "tile.tiled_buffer": A_shared.data}):
                                Out_shared_acc_1[0, ki, kj] = T.float16(0.0)
                    # Accumulation operation between tiles
                    for ki in T.serial(32, annotations={"tile.interior": 1, "tile.interior_axis": 0, "tile.loop_stage": 2, "tile.tiled_buffer": A_shared.data}):
                        for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1, "tile.loop_stage": 2, "tile.tiled_buffer": A_shared.data}):
                            Out_shared_acc_1[0, ki, kj] = Out_shared_acc_1[0, ki, kj] + T.fabs(A_shared[i, j * 32 + ki, k * 32 + kj])
                    if j == 7:
                        # In-tile reduce, result is returned to Out_shared_res
                        T.vector_core_in_tile_reduce("sum", T.region(Out_shared_res[0, 0], 1, 1, 32), T.region(Out_shared_acc_1[0, 0, 0], 1, 1, 32, 32), 0)
                        # when clear=False, wo have to accumulate Out_shared_res to Out_shared data.
                        for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 0, "tile.loop_stage": 2, "tile.tiled_buffer": A_shared.data}):
                            Out_shared[i, k * 32 + kj] = T.fabs(Out_shared[i, k * 32 + kj]) + Out_shared_res[0, kj]
```

## Summary

These tile-related Ops and `T.Tiles()` are ultimately lowered into a unified structure, which is a multi-level loop + (Serial + Vectorized as the two innermost loops) structure. At the same time, sufficient metadata is added to the annotations of each loop, preparing them for the later codegen phase.
