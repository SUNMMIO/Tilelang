# InferLayout and Lowering of the GEMM Operator

## 1 Motivation

GEMM (General Matrix Multiply) is the core operator in deep learning. Our A4E NPU is equipped with dedicated Tensor Cores capable of natively supporting matrix multiplication.
If the native compiler directly generates scalar multiply-accumulate loops, it not only fails to utilize the Tensor Engine but also causes severe execution bottlenecks. To fully exploit the computing power of the NPU, we need to infer specific hardware layouts for the GEMM operator and directly map it to underlying matrix multiply-accumulate instructions (Intrinsics).

## 2 Design

Within the existing operator system of TileLang, we specifically designed a GEMM extension oriented towards the A4E NPU: `kSunmmioMMA`, implemented through `gemm_sunmmio.py` and `src/op/gemm.cc`.
TileLang's GEMM is divided into v1 and v2; v2 is implemented in Python, and v1 is implemented in C++. We provide supports for both.

- **Strict Verification of Storage Scopes:**
  - The NPU's Tensor Engine requires that operands must be stored in specific physical SRAMs. The design enforces that matrix A must be in `shared.asram`, matrix B must be in `shared.wsram`, and matrix C must be in `shared.rsram`.
- **Layout Inference:**
  - Tailored to the access characteristics of the NPU Tensor Engine, the hardware-specific `make_blockwise_zz_layout` (Blockwise ZZ Layout) is forcibly inferred and bound to the three matrices participating in GEMM. This ensures that the computation units achieve optimal bandwidth and alignment efficiency when fetching data.
- **Instruction Lowering:**
  - Completely strips away the traditional high-level scalar loop structures. The high-level `T.gemm` abstract node directly extracts the operand regions and encapsulates them to be lowered into the NPU-specific intrinsic call `tl.mma_sunmmio`, which is then dispatched directly to the backend compiler.

## 3 Implementation

- **Pre-verification and Dispatch:**
  - In `LowerGemm` on the C++ side, the `kSunmmioMMA` instruction type is intercepted.
  - Strict assertions (`ICHECK`) are made on the Scopes of the three Buffers: `A->scope() == "shared.asram"`, `B->scope() == "shared.wsram"`, and `C->scope() == "shared.rsram"`. This forms a perfect closed loop with the `InferSRAMScope` Pass.
- **Layout Inference (`InferLayout`):**
  - On both the Python side (`gemm_sunmmio.py`) and the C++ side, `tl.layout.make_blockwise_zz_layout` is invoked on the input A, B, and C Buffers. The compiler will automatically apply this transformation during memory allocation.
- **Instruction Lowering (`Lower`):**
  - On the Python side, `buffer_region_to_tile_region` is used to convert the input parameters. Similar to Copy, the buffers are transformed into BufferRegions here. This design fixes the parameter length, facilitating subsequent processing. To conveniently capture the GEMM later, the GEMM is also packed into a block named `_gemm_sss`.
  - The parameters are `[A_region, B_region, C_region, transA, transB, clearAccum]`, where the first three are BufferRegions and the latter three are boolean types.

## 4 Example

**Before GEMM Extension：**

```python
# High-level GEMM abstraction with implicit scalar loop semantics
T.gemm(A_shared[0:8, 16:32], B_shared[0:16, 8:16], C_shared[8:24, 16:32], transpose_A=True, transpose_B=True)
```

**After GEMM Extension (InferLayout & Lower)：**

```python
# 1. Layout Inference: blockwise_zz_layout is implicitly applied to A, B, and C

# 2. Instruction Lowering: Strips scalar loops and directly emits hardware MMA instructions
with T.block("_gemm_sss"):
    T.reads()
    T.writes()
    T.mma_sunmmio(T.region(A_shared[0, 16], 1, 8, 16), T.region(B_shared[0, 8], 1, 16, 8), T.region(C_shared[8, 16], 3, 16, 16), T.bool(True), T.bool(True), T.bool(False))
```
