
# InferSRAMScope

## 1 Motivation

In NPU-oriented programming, due to the lack of a unified hardware Cache, all memory must be explicitly managed by software. Native TileLang/TVM only defines memory scopes such as `local` or `shared` at a high level of abstraction. However, our NPU possesses a complex memory hierarchy, where SRAM is specifically divided into three types: ASRAM, WSRAM, and RSRAM. We need to explicitly allocate an accurate scope for each Buffer, which is essential for subsequent passes, such as legality checks.

## 2 Design

We designed and implemented a new Pass, `InferSRAMScope`, to automatically infer the scope for all Buffers. The core logic is as follows:

- **Traverse the AST to collect all shared buffers and contexts**
  - Buffers can fall into two categories: those whose scopes have already been explicitly specified by the user, and those waiting for compiler inference.

- **Infer scopes based on hard constraints**
  - The GEMM operator is temporarily the only hard constraint. The matrix A of GEMM corresponds to `shared.asram`, matrix B to `shared.wsram`, and matrix C to `shared.rsram`. If conflicts occur during this process, the compiler will report an error.

- **Infer scopes based on soft constraints**
  - Under the premise that all hard constraints are satisfied, the compiler will infer the remaining unspecified Buffers to be `shared.rsram`.

- **Globally rewrite the AST based on the inferred scopes**
  - Due to the immutability of TVM IR, the properties of objects cannot be directly modified; they can only be rewritten.

## 3 Implementation

This Pass inherits from `arith::IRMutatorWithAnalyzer`, and its implementation mechanism strictly corresponds to the design logic in Section 1.2:

- **Traverse the AST to collect shared buffers and contexts**
  - By overloading the `StmtExprMutator::VisitStmt_()` family of functions, the entire AST is traversed with `replace_flag = false`, indicating that information is currently being collected and no actual replacement is performed.

- **Infer scopes based on hard constraints**
  - In `VisitStmt_(const EvaluateNode*)`, the `tl.tileop.gemm` node is specifically intercepted and parsed.
  - The three operand Regions of GEMM are extracted: the Buffer corresponding to input matrix A is mapped to `shared.asram`, input matrix B to `shared.wsram`, and output matrix C to `shared.rsram`.
  - Before writing into the `buffer_remap_` dictionary, a strict verification is performed to check whether the Buffer has already been assigned a conflicting scope. If so, an `ICHECK` error is triggered.

- **Infer scopes based on soft constraints**
  - After the first pass of traversal is completed, the fallback function `InferUnspecifiedBuffer()` is called.
  - It iterates through the pending collection of Buffers that have not been constrained by the GEMM operator, uniformly infers their default scopes as `shared.rsram`, and updates the mapping dictionary.

- **Globally deep rewrite the IR**
  - **Object Reconstruction:** It relies on the core helper function `makeBufferWithScope`. This function extracts the pointer type of the original Buffer, creates a new `PointerType` (with the inferred `storage_scope`), and accordingly generates entirely new `Var` and `Buffer` objects, maintaining an injective mapping from the old objects to the new ones.
  - **Attribute Updating:** In the second traversal with `replace_flag = true`, within the `BlockRealizeNode`, not only is the `alloc_buffers` list replaced, but the Block's `annotations` (such as `kLayoutMap` and `kTileViewMap`) are also deeply rewritten because the keys of these dictionaries strongly depend on the old Buffer Var.
  - **Expression Replacement:** It traverses all `BufferLoad`, `BufferStore` nodes, and external calls like `tvm_access_ptr`, thoroughly replacing the old Buffers and pointer Handles referenced within them with the new objects, thereby completing the legalization of the IR.

## 4 Example

**Before InferSRAMScope：**

```python
A = T.alloc_shared((128, 128), dtype="float16")
B = T.alloc_shared((128, 128), dtype="float16")
C = T.alloc_shared((128, 128), dtype="float16")
D = T.alloc_shared((128, 128), dtype="float16")
T.gemm(A, B, C)
```

**After InferSRAMScope：**

```python
A = T.alloc_shared((128, 128), dtype="float16", scope="shared.asram")
B = T.alloc_shared((128, 128), dtype="float16", scope="shared.wsram")
C = T.alloc_shared((128, 128), dtype="float16", scope="shared.rsram")
D = T.alloc_shared((128, 128), dtype="float16", scope="shared.rsram")
T.gemm(A, B, C)
```
