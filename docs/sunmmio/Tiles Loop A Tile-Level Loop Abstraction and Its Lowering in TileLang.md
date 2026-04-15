# Introducing Tiles Loop: A Tile-Level Loop Abstraction and Its Lowering in TileLang

## 1. Introduction
### 1.1 Motivation
Driven by the evolving landscape of modern AI compilers and the specific architectural constraints of our proprietary hardware (particularly the rigid data locality and alignment requirements of the vector core), we established a strategic objective: **elevating the "Tile" to a First-Class Citizen within TileLang-Mesh**. This necessitates providing robust, native representation and computational support at both the frontend Domain-Specific Language (DSL) and the compiler backend.

Since TileLang is built atop the TVM stack, it originally lacked an explicit, concise data structure or loop abstraction for tile-level semantics. Consequently, we architected a native solution. Ultimately, our goal is to empower users with an intuitive and expressive syntax for tile-level operations, as demonstrated below:

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), T.ceildiv(Batch, block_B), threads=128) as (bx, by, bz):
    A_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
    # Partition tensors A, B, and C according to the specified tile_size
    T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})

    B_shared = T.alloc_shared((block_B, block_M, block_N), dtype)
    T.annotate_tileview({B_shared: make_tileview(B_shared, tile_size, index_map)})

    C_shared = T.alloc_shared((block_B, block_M, block_N), accum_dtype)

    T.copy(A[bz * block_B:(bz + 1) * block_B, by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N], A_shared)
    T.copy(B[bz * block_B:(bz + 1) * block_B, by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N], B_shared)

    # Utilize the T.Tiles() primitive to represent tile-level parallel execution
    for b, i, j in T.Tiles(A_shared, parallel=True):
        # Within the T.Tiles() scope, A_shared[b, i, j], B_shared[b, i, j], and C_shared[b, i, j]
        # represent a discrete, fine-grained tile block rather than a single scalar element.
        # Tile operations map naturally to element-wise vectorized instructions.
        C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]
        C_shared[b, i, j] = A_shared[b, i, j] * T.float32(2.0)
        C_shared[b, i, j] = T.exp(A_shared[b, i, j]) + T.exp(B_shared[b, i, j])

    ...
```

---

## 2. Implementation Architecture
### 2.1 Design Rationale
As previously noted, TileLang natively lacked a dedicated "Tile Loop" AST node. Constructing a completely novel loop type from scratch within the existing TVM/TileLang IR would introduce immense engineering overhead and pose a high risk of destabilizing the intricate, pre-existing compiler pass pipeline.

To mitigate this, we architected a pragmatic lowering strategy: mapping the `T.Tiles()` abstraction to a composite **`Serial Loop + Vectorized Loop`** structure. Because both `Serial` and `Vectorized` loop nodes are first-class constructs already supported by TileLang, this approach seamlessly integrates with the existing infrastructure. Furthermore, encapsulating computations within an innermost `Vectorized` loop inherently unlocks support for a vast array of existing scalar arithmetic operations, guaranteeing excellent operator completeness without redundant engineering.

### 2.2 The Lowering Pipeline
The end-to-end implementation of the `T.Tiles()` primitive spans the frontend DSL and is progressively lowered through several critical backend passes:

1. **Frontend DSL Extensions (Modifications to `ir.cc`, etc.)**:
   - We extended the AST builder to emit a preliminary multi-level serial loop nest corresponding to the tile grid. These loops are disambiguated from standard serial loops via specialized internal fields/annotations.
2. **`LegalizeTilesLoop` Pass (New)**:
   - This newly introduced pass is responsible for extracting the `TileView` metadata (the tensor partitioning schema). It subsequently attaches this metadata as annotations to the corresponding loop nodes and dynamically recalculates the iteration extents of the loop variables based on the spatial mapping defined by the Tile. At this juncture, the IR acquires preliminary tile semantics.
3. **`TilesLoop` Pass (New)**:
   - This core pass executes the structural expansion. It injects two inner loop layers—an outer `Serial` loop and an innermost `Vectorized` loop—at the leaf of the previously legalized loop nest, effectively materializing the dense computation within a single tile.
   - Concurrently, it injects essential low-level annotations (e.g., `"tile.scope_entry"`, `"tile.interior"`) and performs index substitution on the buffer accesses within the loop body, mapping the abstract logical indices strictly to physical memory coordinates.
4. **`loop_vectorize` Pass Adjustments**:
   - We augmented the existing vectorization pass with target-specific branching. This includes tuning vectorization heuristics and parameters (e.g., maximum vector length) to precisely align with the hardware characteristics of our Sunmmio vector core.

---

## 3. Lowering Results Walkthrough

The following IR snippets demonstrate the progressive transformation of the `T.Tiles()` abstraction through the compiler pipeline.

### 3.1 Step 1: Initial Frontend IR Construction
At the DSL level, `T.Tiles()` is initially represented as a standard nested serial loop iterating over the tile grid, tagged with basic annotations.

```python
for b in T.serial(16, annotations={"tile.loop_parallel": 1, "tile.loop_stage": 0, "tile.tiled_buffer": A_shared}):
    for i in T.serial(256, annotations={"tile.loop_parallel": 1, "tile.loop_stage": 0, "tile.tiled_buffer": A_shared}):
        for j in T.serial(128, annotations={"tile.loop_parallel": 1, "tile.loop_stage": 0, "tile.tiled_buffer": A_shared}):
            C_shared[b, i, j] = A_shared_1[b, i, j] * B_shared_1[b, i, j]
            C_shared[b, i, j] = T.Cast("float16", T.Cast("float32", A_shared_1[b, i, j]) * T.float32(2.0))
            C_shared[b, i, j] = T.exp(A_shared_1[b, i, j]) + T.exp(B_shared_1[b, i, j])
```

### 3.2 Step 2: Post `LegalizeTilesLoop` Pass
The compiler resolves the `TileView` mapping. The iteration extents are shrunk to represent the number of tiles rather than scalar elements, and richer annotations (`tile.buffer_new_shape`, `tile.dim_map`) are populated.

```python
# Annotations are enriched and iteration ranges are downscaled to reflect tile coordinates
for b in T.serial(16, annotations={"tile.buffer_new_shape": [16, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.loop_parallel": 1, "tile.loop_stage": 1, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
    for i in T.serial(8, annotations={"tile.buffer_new_shape": [16, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.execution": 1, "tile.loop_parallel": 1, "tile.loop_stage": 1, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
        for j in T.serial(4, annotations={"tile.buffer_new_shape": [16, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.execution": 1, "tile.loop_parallel": 1, "tile.loop_stage": 1, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
            C_shared[b, i, j] = A_shared[b, i, j] * B_shared[b, i, j]
            C_shared[b, i, j] = T.Cast("float16", T.Cast("float32", A_shared[b, i, j]) * T.float32(2.0))
            C_shared[b, i, j] = T.exp(A_shared[b, i, j]) + T.exp(B_shared[b, i, j])
```

### 3.3 Step 3: Post `TilesLoop` Pass
The pass expands the innermost body by injecting the dense execution loops (`ki` and `kj`). Buffer indices within the block are re-mapped to address the specific spatial elements inside the current tile.

```python
for b in T.serial(16, annotations={"tile.buffer_new_shape": [16, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.loop_parallel": 1, "tile.loop_stage": 2, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
    for i in T.serial(8, annotations={"tile.buffer_new_shape": [16, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.execution": 1, "tile.loop_parallel": 1, "tile.loop_stage": 2, "tile.scope_entry": 1, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):
        for j in T.serial(4, annotations={"tile.buffer_new_shape": [16, 8, 4, 32, 32], "tile.dim_map": [-2, -1], "tile.execution": 1, "tile.loop_parallel": 1, "tile.loop_stage": 2, "tile.tile_size": [32, 32], "tile.tiled_buffer": A_shared.data}):

            # The two inner layers representing the physical tile execution are injected.
            # The structured presence of these annotations acts as a definitive signature
            # for the backend Codegen to dispatch to vector instructions.
            for ki in T.serial(32, annotations={"tile.interior": 1, "tile.interior_axis": 0, "tile.loop_stage": 2, "tile.tiled_buffer": A_shared.data}):
                for kj in T.vectorized(32, annotations={"tile.interior": 1, "tile.interior_axis": 1, "tile.loop_stage": 2, "tile.tiled_buffer": A_shared.data}):

                    # Buffer indices are dynamically rewritten to reflect the exact physical memory offset
                    C_shared[b, i * 32 + ki, j * 32 + kj] = A_shared[b, i * 32 + ki, j * 32 + kj] * B_shared[b, i * 32 + ki, j * 32 + kj]
                    C_shared[b, i * 32 + ki, j * 32 + kj] = T.Cast("float16", T.Cast("float32", A_shared[b, i * 32 + ki, j * 32 + kj]) * T.float32(2.0))
                    C_shared[b, i * 32 + ki, j * 32 + kj] = T.exp(A_shared[b, i * 32 + ki, j * 32 + kj]) + T.exp(B_shared[b, i * 32 + ki, j * 32 + kj])
```

---

## 4. Summary
Following the execution of these passes, the structural and computational semantics of the `T.Tiles()` abstraction are fully materialized. By anchoring the design on a `Serial + Vectorized` loop foundation, the internal block of a `T.Tiles()` construct effortlessly inherits support for comprehensive element-wise operations (e.g., `+`, `-`, `*`, `/`, `exp`, `log`, `sin`, `sqrt`, `abs`).

Ultimately, this pipeline emits a highly predictable, structurally pristine Intermediate Representation (IR) embedded with rich metadata annotations, ensuring a frictionless transition into the subsequent Codegen phase for optimal hardware-specific instruction emission.
