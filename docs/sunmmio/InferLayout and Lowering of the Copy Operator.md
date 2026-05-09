
# InferLayout and Lowering of the Copy Operator

## 1 Motivation

Our A4E NPU is equipped with a dedicated DMA engine. To unleash hardware performance, the compiler must deduce specific Layouts (such as the Blockwise ZZ Layout) that satisfy hardware alignment requirements based on the source and destination storage scopes of data movement (e.g., from `global` to `shared.asram`, or within `shared.rsram`). Furthermore, the high-level Copy operator must be precisely lowered into corresponding asynchronous DMA instructions or loops annotated with `tile_level_loop`.

## 2 Design

Building upon the original `copy.cc`, two core copy modes specifically tailored for the A4E NPU have been extended: `kSunmmioDMACopy` and `kSunmmioTileCopy`.

- **Mode Recognition and Dispatch:**
  - When the movement crosses storage scopes (e.g., `global <-> shared.*`) and the data types match, it branches into `kSunmmioDMACopy`.
  - When the movement is `shared.rsram -> shared.rsram` and it is a full buffer copy, it branches into `kSunmmioTileCopy`.

- **Layout Inference:**
  - For these two modes, if the destination/source Buffer is located on-chip and has dimensions > 1, the compiler will automatically infer and allocate a hardware-specific ZZ layout to avoid bank conflicts and adapt to subsequent computations. Developer-facing code should use `make_zz_layout`; compiler inference constructs the same layout in C++ through `sunmmio::MakeZZ` and gets the target/dtype-specific block shape from `sunmmio_utils`.

- **Instruction Lowering:**
  - For DMA Copy, it is directly lowered into the intrinsic call of `tl.dma_copy`.
    - `tl.dma_copy` is defined as: `dma_copy(src_region, dst_region)`, where both `src_region` and `dst_region` are of type `RegionOp`. They describe not only the storage scope of the data but also the data range to be moved. This design also fixes the parameter length; regardless of the dimensionality of the `RegionOp`, there are always exactly two parameters, facilitating subsequent processing.
  - For Tile Copy, it is lowered into a serial loop annotated with `tile_level_loop`, which is utilized by the subsequent TilesLoop Pass.

## 3 Implementation

The core logic is implemented in `src/op/copy.cc` and is divided into three stages: trigger condition checking, layout inference, and code generation:

- **Trigger Condition Checking (`CheckSunmmioDMACopy` & `CheckSunmmioTileCopy`)**
  - `CheckSunmmioDMACopy`: Strictly verifies whether the `storage_scope` combination of `src` and `dst` complies with DMA movement rules and requires the data types to be strictly equal.
  - `CheckSunmmioTileCopy`: In addition to restricting the scopes to be `shared.rsram`, it also ensures that it is a duplication of the entire buffer through `is_full_buffer_copy`.

- **Layout Inference (`InferLayout`)**
  - At the `InferLevel::kFree` level, the Copy node is intercepted. If the buffer has not yet been bound to a layout and is not in the global scope, the compiler remaps the buffer to a ZZ layout, ensuring that the memory access addresses generated later conform to the NPU's ZZ Layout.

- **Instruction Lowering (`LowerSunmmioDmaCopy` & `LowerSunmmioTileCopy`)**
  - **DMA Lowering**: Calls `MakeRegionExpr` to pack the original multi-dimensional slice accesses into Region expressions, and then emits `T.call_intrin("tl.dma_copy", src_region, dst_region)`.
  - **Tile Lowering**: Constructs a standard nested `For` loop to execute read and write operations. The key point is to attach metadata such as `tile_level_loop`, `tiled_buffer`, and `kTileLoopStage` to the generated loop nodes for use by subsequent Passes.

## 4 Example

**Before：**

```python
# Simple logical copy, no Layout specified
T.copy(A_global[0:128, 0:128], A_rsram[0:128, 0:128])
T.copy(A_shared, B_shared)
```

**After：**

```python
# 1. global -> rsram triggers kSunmmioDMACopy
# Inferred as ZZ layout, and lowered to DMA intrinsic
T.dma_copy(T.region(A_global[0, 0], 1, 128, 128), T.region(A_rsram[0, 0], 2, 128, 128))

# 2. rsram -> rsram triggers kSunmmioTileCopy
# Inferred as ZZ layout, and generates a loop annotated with Tile attributes
for v0 in T.serial(64, annotations={"tile.loop_parallel": 1, "tile.loop_stage": 0, "tile.tiled_buffer": A_shared.data}):
    for v1 in T.serial(64, annotations={"tile.loop_parallel": 1, "tile.loop_stage": 0, "tile.tiled_buffer": A_shared.data}):
        B_shared[v0, v1] = A_shared[v0, v1]
```
