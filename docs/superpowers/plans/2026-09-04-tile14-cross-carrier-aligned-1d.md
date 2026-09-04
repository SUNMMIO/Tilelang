# TILE14 Cross-Carrier Aligned-1D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve a complete rank-1 `T.Tiles([14])` execution tile for BF16 and FP32 flat aligned-row-major RSRAM buffers, lowering logical windows that may cross a 64-byte carrier through two legal carrier views.

**Architecture:** The T.Tiles planner locally injects one relaxed candidate equal to the complete static rank-1 domain, without changing shared TileView enumeration or 2D fallback. Sunmmio codegen keeps every physical view at 64 bytes, emits independent runtime single/double-carrier branches for each load and store, and conservatively invalidates destination data-cache entries after a cross-capable store.

**Tech Stack:** TileLang C++, TVM TIR, Sunmmio SUVM MLIR, pytest, NPU-IR `npuir-opt`, A4E ELF toolchain

**Spec:** `/workspace/tilelang-samples/my_sample/minimax_m3_vl/TILE14_ALIGNED_1D_BRIDGE_DESIGN.md`

## Global Constraints

- Only original static rank-1 `T.Tiles` domains may receive the new non-divisor candidate; rank-reduction search from an original 2D domain remains unchanged.
- The new candidate supports only BF16 and FP32 RSRAM buffers with an explicit flat aligned-row-major `CuteLayout`.
- The tiled dimension must be the trailing stride-1 dimension; its covered extent and every outer stride must be divisible by the 64-byte carrier extent.
- Manual TileViews, ZZ/hierarchical layouts, FP16, integer, FP8, MX, sub-byte, reduction singleton writeback, small-2D fallback, and scalar fallback retain current behavior.
- Physical memory accesses remain `tile_view<32xbf16>` or `tile_view<16xf32>`; a double-carrier type is a register tile, never a tile view.
- Source and destination crossings are lowered independently.
- A cross-capable store invalidates `current_tile_values` for its destination buffer after the branch; `tile_view_cache` is retained.
- Current pinned NPU-IR is used during implementation. Latest NPU-IR and gem5 correctness/performance are deferred to the unified final validation phase.

---

### Task 1: Preserve the Complete Rank-1 Domain in the Planner

**Files:**
- Modify: `testing/python/sunmmio/transform/test_infer_tileview.py`
- Modify: `src/tileview/tileview_planner.cc`

**Interfaces:**
- Consumes: existing `AddRank1Candidate`, `AnalyzeAccessesForExecutionAxes`, `SupportsAligned1DBridgeCandidate`, and `TileViewPlan::requires_aligned_1d_bridge`.
- Produces: `SupportsCrossCarrierAligned1DBridgeCandidate(...)` and an optional complete-domain extent passed only by the original rank-1 planning call.

- [ ] **Step 1: Add the BF16/FP32 planner test and scope guards**

Add a parameterized kernel whose logical storage has 28 elements and whose aligned covered width is 32. The literal expected execution tile is 14:

```python
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_infer_complete_rank1_nondivisor_domain(dtype):
    logical_width = 28
    tile_width = 14

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((logical_width,), dtype, scope="shared.rsram")
            dst = T.alloc_shared((logical_width,), dtype, scope="shared.rsram")
            layout = make_aligned_row_major(
                (logical_width,), dtype, align_bytes=64
            )
            T.annotate_layout({src: layout, dst: layout})
            for j in T.Tiles([tile_width]):
                dst[j] = src[j]

    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = IRModule.from_expr(
        main.with_attr("global_symbol", "main").with_attr("target", target)
    )
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, [14], [0])
```

This test catches removal of the complete-domain candidate or accidental reintroduction of covered-width divisor-only planning.

Add three literal counterexamples in the same file:

```python
def test_fp16_nondivisor_rank1_keeps_legacy_plan():
    logical_width = 28

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((logical_width,), "float16", scope="shared.rsram")
            dst = T.alloc_shared((logical_width,), "float16", scope="shared.rsram")
            layout = make_aligned_row_major(
                (logical_width,), "float16", align_bytes=64
            )
            T.annotate_layout({src: layout, dst: layout})
            for j in T.Tiles([14]):
                dst[j] = src[j]

    mod = lower_for_sunmmio(main)
    assert_scope_plan(mod, [2], [0])


def test_zz_nondivisor_rank1_keeps_legacy_plan():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            dst = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            layout = make_zz_layout((32, 32), [0, 1], (32, 32))
            T.annotate_layout({src: layout, dst: layout})
            for j in T.Tiles([14]):
                dst[0, j] = src[0, j]

    mod = lower_for_sunmmio(main)
    assert_scope_plan(mod, [2], [0])


def test_rank2_domain_does_not_enable_nondivisor_side_bridge():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            matrix = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            side = T.alloc_shared((28,), "bfloat16", scope="shared.rsram")
            T.annotate_layout(
                {
                    matrix: make_zz_layout((32, 32), [0, 1], (32, 32)),
                    side: make_aligned_row_major((28,), "bfloat16", align_bytes=64),
                }
            )
            for i, j in T.Tiles([4, 14]):
                matrix[i, j] = matrix[i, j] + side[j]

    mod = lower_for_sunmmio(main)
    tile_size, _ = get_single_scope_plan(mod)
    assert tile_size != [4, 14]
```

Implement `lower_for_sunmmio` and `get_single_scope_plan` next to the existing
`assert_scope_plan` traversal so all four tests use the same target attachment
and `LowerTilesLoop` path. The guard tests are characterization tests: they may
already pass before the production change, but they must be present before it.

- [ ] **Step 2: Run the planner test and verify RED**

Run:

```bash
pytest -q testing/python/sunmmio/transform/test_infer_tileview.py \
  -k complete_rank1_nondivisor_domain
```

Expected: both positive cases fail because the current planner selects `[2]`,
not `[14]`. Run the three guard tests separately and record their baseline;
they must continue to reject the new `[14]` plan.

- [ ] **Step 3: Add explicit cross-carrier support predicates**

Keep `SupportsAligned1DBridgeCandidate` as the legacy path. Add a separate predicate whose logic is equivalent to:

```cpp
bool SupportsCrossCarrierAligned1DBridgeDType(DataType dtype) {
  return dtype.is_bfloat16() || (dtype.is_float() && dtype.bits() == 32);
}

bool SupportsCrossCarrierAligned1DBridgeCandidate(
    const AccessTileCandidate &candidate, const Buffer &buffer,
    const Map<Buffer, Layout> &layout_map,
    const SunmmioTileProcessorConfig &config) {
  // Require rank 1, RSRAM, 0 < L < C, BF16/FP32, an explicit flat
  // row-major CuteLayout, trailing stride-1 mapping, carrier-padded covered
  // width, and carrier-aligned outer strides.
}
```

Use `candidate.tileview->IndexMap()` to confirm the mapped dimension is the last logical buffer dimension. Inspect `CuteLayoutNode::GetDimLevels()`, `GetCoveredShape()`, and `GetModeStrideOfDim()` rather than reconstructing layout from strings.

- [ ] **Step 4: Inject the complete-domain candidate only for original rank 1**

Extend `AnalyzeAccessesForExecutionAxes` with:

```cpp
std::optional<int> cross_carrier_domain_extent
```

The full-rank call in `TryPlanTileViewsForTilesScope` supplies `domain[0]` only when `domain.size() == 1` and the extent is a positive `IntImm`. The original-2D rank-reduction calls pass `std::nullopt`.

For an eligible inferred access with exactly one active binding, call:

```cpp
AddRank1Candidate(&relaxed_candidates, access.buffer, access.indices,
                  bindings, mapped_dim, *cross_carrier_domain_extent,
                  analyzer, /*strict_checks=*/false);
```

Retain that candidate only if either the legacy bridge predicate accepts it or `SupportsCrossCarrierAligned1DBridgeCandidate` accepts it under the enabled context. Deduplicate equal `tile_shape`/index-map candidates before plan selection.

- [ ] **Step 5: Rebuild and verify GREEN**

Run from `/workspace/tilelang-samples`:

```bash
make build-tilelang
```

Then run the focused planner test. Expected: BF16 and FP32 both select `[14]`.

- [ ] **Step 6: Re-run the planner scope guards**

Run the three tests added in Step 1 and confirm FP16, hierarchical layout, and
the original-rank-2 search still reject the non-divisor `[14]` plan.

- [ ] **Step 7: Run planner regression and commit**

Run:

```bash
pytest -q testing/python/sunmmio/transform/test_infer_tileview.py
git diff --check
git add src/tileview/tileview_planner.cc \
  testing/python/sunmmio/transform/test_infer_tileview.py
git commit -m "feat(sunmmio): plan complete nondivisor rank1 tiles"
```

Expected: planner suite passes and only the two listed files are committed.

---

### Task 2: Lower Cross-Carrier Rank-1 Loads

**Files:**
- Modify: `testing/python/sunmmio/codegen/test_tiles_aligned_store.py`
- Modify: `src/target/sunmmio/sunmmio_codegen_tiles_loop.cc`

**Interfaces:**
- Consumes: the planner's `[14]` plan; existing `Aligned1DAddressInfo`, `make_tile_cache_key`, `current_tile_values`, `TileSlice`, `TileInsertSlice`, and `BeginIf` live-out support.
- Produces: `TileAccessInfo::may_cross_aligned_1d_carrier`, `next_aligned_1d_address(...)`, `load_aligned_1d_carrier(...)`, `combine_aligned_1d_carriers(...)`, and a cross-capable load branch returning `tile<L>`.

- [ ] **Step 1: Add a dynamic-offset TIR fixture**

In `test_tiles_aligned_store.py`, add a low-level annotated TIR builder parameterized by `dtype`. Its outer serial loop has extent 4; its inner tile scope has domain/tile size 14; source and destination index is `segment * 14 + lane`; both buffers use `make_aligned_row_major((64,), dtype, 64)`.

The fixture must produce the same annotated structure as existing `_make_nonzero_offset_aligned_store_stmt`, with:

```text
tile.domain = [14]
tile.tile_size = [14]
tile.execution_domain_axes = [0]
```

and a dynamic `segment` outside the tile scope so the carrier remainder cannot be constant-folded before codegen.

- [ ] **Step 2: Add the failing load structure test**

```python
@pytest.mark.parametrize(
    "dtype,carrier,wide,mlir_dtype",
    [
        ("bfloat16", 32, 64, "bf16"),
        ("float32", 16, 32, "f32"),
    ],
)
def test_cross_carrier_rank1_load_uses_runtime_carrier_window(
    dtype, carrier, wide, mlir_dtype, tmp_path
):
    src = _build_sunmmio_source_from_func(
        _make_cross_carrier_rank1_func(dtype=dtype)
    )
    validate_suvm_mlir_with_npuir_opt(
        src, tmp_path,
        mlir_filename=f"rank1_cross_load_{dtype}.mlir",
        opt_args=("--verify-each",),
    )
    assert f"!suvm.tile_view<{carrier}x{mlir_dtype}>" in src
    assert f"!suvm.tile<{wide}x{mlir_dtype}>" in src
    assert f"!suvm.tile_view<{wide}x{mlir_dtype}>" not in src
    assert "scf.if" in src
    assert f"][14] : !suvm.tile<{wide}x{mlir_dtype}>" in src
```

The production change that makes this pass is the register-only double-carrier load branch; a single-carrier dynamic slice cannot satisfy the wide-tile assertion.

- [ ] **Step 3: Run the load test and verify RED**

Run:

```bash
pytest -q testing/python/sunmmio/codegen/test_tiles_aligned_store.py \
  -k cross_carrier_rank1_load
```

Expected: both cases fail because no wide register tile or carrier-crossing `scf.if` is emitted.

- [ ] **Step 4: Classify cross-capable aligned accesses**

Add to `TileAccessInfo`:

```cpp
bool may_cross_aligned_1d_carrier{false};
```

Set it in `populate_aligned_1d_access` when:

```cpp
access->requires_aligned_1d_load &&
access->aligned_load_elems % access->tile_shape[0] != 0
```

Before enabling it, validate BF16/FP32 and the flat row-major carrier-alignment invariants from the spec against the bound `SunMMIOType`. Emit an `ICHECK` diagnostic if hand-authored TIR bypasses the planner contract.

- [ ] **Step 5: Extract carrier helpers**

Implement internal lambdas with these contracts:

```cpp
Aligned1DAddressInfo next_aligned_1d_address(
    const TileAccessInfo &access, const Aligned1DAddressInfo &base);

SunMMIOValue load_aligned_1d_carrier(
    const TileAccessInfo &access, const Aligned1DAddressInfo &address,
    TileBlockState *state);

SunMMIOValue combine_aligned_1d_carriers(
    const TileAccessInfo &access, const SunMMIOValue &c0,
    const SunMMIOValue &c1);
```

`load_aligned_1d_carrier` owns the 64-byte view construction and current-value cache lookup. `combine_aligned_1d_carriers` creates a zero `tile<2C>` and inserts `c0` at offset 0 and `c1` at offset C.

- [ ] **Step 6: Emit the load `scf.if` result**

In `load_aligned_1d_tile`, retain the legacy body when `may_cross_aligned_1d_carrier` is false. Otherwise:

```cpp
SunMMIOValue cross = builder_->Compare(
    NewValueName(), CompareOp::kGT, CompareDomain::kSignedInt,
    add_index(address.offset_elems, make_index_const(L)),
    make_index_const(C), index_type);

SunMMIOValue initial = make_zero_tile(logical_type, value_dtype);
SunMMIOValue result = builder_->BindValueAlias(NewValueName(), initial);
builder_->BeginIf(cross, std::vector<SunMMIOValue>{result});
SunMMIOValue c1 = load_aligned_1d_carrier(
    access, next_aligned_1d_address(access, address), state);
SunMMIOValue wide = combine_aligned_1d_carriers(access, c0, c1);
SunMMIOValue cross_slice = builder_->TileSlice(
    NewValueName(), wide, {address.offset_elems}, {L});
builder_->BindValueAlias(result.name, cross_slice);
builder_->BeginElse();
SunMMIOValue fast_slice = builder_->TileSlice(
    NewValueName(), c0, {address.offset_elems}, {L});
builder_->BindValueAlias(result.name, fast_slice);
builder_->EndIf();
return result;
```

Do not emit the single-carrier slice before `BeginIf`; it is out of bounds for crossing runtime values.

- [ ] **Step 7: Rebuild, verify GREEN, and commit**

Run:

```bash
make -C /workspace/tilelang-samples build-tilelang
pytest -q testing/python/sunmmio/codegen/test_tiles_aligned_store.py \
  -k cross_carrier_rank1_load
git diff --check
git add src/target/sunmmio/sunmmio_codegen_tiles_loop.cc \
  testing/python/sunmmio/codegen/test_tiles_aligned_store.py
git commit -m "feat(sunmmio): load rank1 tiles across rsram carriers"
```

Expected: BF16 and FP32 raw SUVM verifies and contains only 64-byte physical views.

---

### Task 3: Lower Stores, Predicates, and Conservative Cache Invalidation

**Files:**
- Modify: `testing/python/sunmmio/codegen/test_tiles_aligned_store.py`
- Modify: `testing/python/sunmmio/codegen/test_tiles_fallback_opt_validate.py`
- Modify: `src/target/sunmmio/sunmmio_codegen_tiles_loop.cc`

**Interfaces:**
- Consumes: Task 2 carrier helpers and `TileAccessInfo::may_cross_aligned_1d_carrier`.
- Produces: `store_cross_carrier_aligned_1d_tile(...) -> void`; existing `store_aligned_1d_tile(...) -> SunMMIOValue` remains the legacy/singleton path.

- [ ] **Step 1: Add failing store structure, predicate, and strict-pipeline tests**

Extend `_make_cross_carrier_rank1_func` with `explicit_predicate=False`. For the predicate variant attach `lane < 13` to both the source `BufferLoad` and destination `BufferStore`.

Add tests asserting:

```python
assert src.count("scf.if") >= 2  # independent source and destination branches
assert "suvm.tile.insert_slice" in src
assert src.count("suvm.tile.store") >= 3  # two in cross branch, one in fast branch
assert f"!suvm.tile_view<{wide}x{mlir_dtype}>" not in src
```

For the predicate variant additionally assert `suvm.tile.select` is present and its result type is the logical `!suvm.tile<14x...>` rather than a carrier type.

In `test_tiles_fallback_opt_validate.py`, add a tracked BF16/FP32 kernel with
the same four serial 14-element segments and aligned 64-element source and
destination buffers. Validate it with:

```python
STRICT_OPT_ARGS = ("--verify-each", "--suvm-to-llvm-pipeline")

@pytest.mark.parametrize(
    "dtype,carrier,wide,mlir_dtype",
    [
        (T.bfloat16, 32, 64, "bf16"),
        (T.float32, 16, 32, "f32"),
    ],
)
def test_cross_carrier_rank1_full_pipeline(
    dtype, carrier, wide, mlir_dtype, tmp_path
):
    source = validate_sunmmio_codegen_with_npuir_opt(
        cross_carrier_rank1_kernel(dtype),
        tmp_path,
        mlir_filename=f"rank1_cross_pipeline_{mlir_dtype}.mlir",
        expected_tokens=(
            f"!suvm.tile_view<{carrier}x{mlir_dtype}>",
            f"!suvm.tile<{wide}x{mlir_dtype}>",
            "scf.if",
        ),
        opt_args=STRICT_OPT_ARGS,
    )
    assert f"!suvm.tile_view<{carrier}x{mlir_dtype}>" in source
    assert f"!suvm.tile<{wide}x{mlir_dtype}>" in source
    assert f"!suvm.tile_view<{wide}x{mlir_dtype}>" not in source
```

- [ ] **Step 2: Run store tests and verify RED**

Run:

```bash
pytest -q testing/python/sunmmio/codegen/test_tiles_aligned_store.py \
  -k 'cross_carrier_rank1 and (store or predicate)'
pytest -q testing/python/sunmmio/codegen/test_tiles_fallback_opt_validate.py \
  -k cross_carrier_rank1
```

Expected: failure because the destination still uses one carrier insert/store
and lacks a separate store branch. The strict pipeline test also remains red
until the complete load/store control flow is legal.

- [ ] **Step 3: Implement cross-carrier store RMW**

Add:

```cpp
void store_cross_carrier_aligned_1d_tile(
    const TileAccessInfo &access, const SunMMIOValue &value,
    const std::optional<SunMMIOValue> &store_mask,
    TileBlockState *state);
```

The lambda implementation must:

```text
load/reuse c0
compute cross = r + L > C
then:
  load/reuse c1
  combine c0/c1 into tile<2C>
  normalize mask to tile<L> and select(value, old logical slice)
  insert logical tile at dynamic r
  extract fixed [0][C] and [C][C]
  store both carrier tiles through separate 64-byte views
else:
  normalize the same mask against slice(c0, r, L)
  insert logical tile into c0
  store one carrier
```

The fast and cross branches share mask-normalization code only where doing so does not create an out-of-bounds slice before the branch.

- [ ] **Step 4: Dispatch without changing reduction singleton stores**

In ordinary `BufferStore` lowering:

```cpp
erase_current_values_for_buffer(state, store->buffer.get());
if (access.may_cross_aligned_1d_carrier) {
  ICHECK(!single_lane_zero_store)
      << "Cross-carrier aligned 1D store does not support singleton writeback";
  store_cross_carrier_aligned_1d_tile(access, rhs, mask, state);
  erase_current_values_for_buffer(state, store->buffer.get());
} else if (access.requires_aligned_1d_load) {
  updated_aligned_tile =
      store_aligned_1d_tile(access, rhs, mask, state,
                            single_lane_zero_store);
}
```

Only the legacy branch republishes one updated carrier into `current_tile_values`. The cross-capable branch leaves no destination data entry after `scf.if`. The reduction-specific call remains on the legacy helper because its logical extent is one and divides every supported carrier.

- [ ] **Step 5: Add a stale-cache regression**

Create a fixture whose tile body stores into a cross-capable destination and then loads that destination for a second output. Assert the raw SUVM contains a destination `suvm.tile.load` after the first store branch. This catches removal of the post-store buffer invalidation, which could otherwise reuse a pre-store carrier alias.

- [ ] **Step 6: Rebuild, verify GREEN, and run aligned-store regression**

Run:

```bash
make -C /workspace/tilelang-samples build-tilelang
pytest -q testing/python/sunmmio/codegen/test_tiles_aligned_store.py \
  -k cross_carrier_rank1
pytest -q testing/python/sunmmio/codegen/test_tiles_aligned_store.py
```

Expected: new BF16/FP32, predicate, and stale-cache cases pass; all existing aligned-1D and small-2D cases remain green.

- [ ] **Step 7: Commit**

```bash
git diff --check
git add src/target/sunmmio/sunmmio_codegen_tiles_loop.cc \
  testing/python/sunmmio/codegen/test_tiles_aligned_store.py \
  testing/python/sunmmio/codegen/test_tiles_fallback_opt_validate.py
git commit -m "feat(sunmmio): store rank1 tiles across rsram carriers"
```

---

### Task 4: Validate the Full Pipeline and MiniMax Integration

**Files:**
- Verify: `testing/python/sunmmio/codegen/test_tiles_fallback_opt_validate.py`
- Modify locally: `/workspace/tilelang-samples/my_sample/minimax_m3_vl/test_conv3d_patchify.py` (excluded by the sample repo's `.git/info/exclude`)
- Verify: `/workspace/tilelang-samples/my_sample/minimax_m3_vl/conv3d_patchify.py`

**Interfaces:**
- Consumes: Tasks 1-3 complete-domain plan and cross-carrier load/store lowering.
- Produces: pinned-NPU-IR regression results and target-kernel plan/compile evidence.

- [ ] **Step 1: Run the strict-pipeline regression added before store implementation**

Run:

```bash
pytest -q testing/python/sunmmio/codegen/test_tiles_fallback_opt_validate.py \
  -k cross_carrier_rank1
```

Expected final state: pinned `npuir-opt` completes the full SUVM-to-LLVM pipeline for both dtypes.

- [ ] **Step 2: Strengthen the MiniMax planner assertion**

Add a helper to the excluded local sample test that runs `LowerTilesLoop` on the final device TIR and records `tile.tile_size`. Add a literal assertion that `conv3d_patchify_serial_tile` contains `[14]` and does not contain `[2]` for its patchify scope.

- [ ] **Step 3: Compile the MiniMax kernel through ELF generation**

Run from `/workspace/tilelang-samples` after sourcing `build/env.sh`:

```bash
source build/env.sh
.venv/bin/python -m pytest -q \
  my_sample/minimax_m3_vl/test_conv3d_patchify.py
```

Then invoke the existing JIT compile path for a reduced compile-only shape and confirm cache artifacts contain:

```text
device_kernel.mlir
kernel.ll
kernel.elf
```

Do not run gem5 in this task.

- [ ] **Step 4: Run focused cross-feature regressions**

Run:

```bash
pytest -q \
  testing/python/sunmmio/transform/test_infer_tileview.py \
  testing/python/sunmmio/codegen/test_tiles_aligned_store.py \
  testing/python/sunmmio/codegen/test_tiles_fallback_opt_validate.py \
  testing/python/sunmmio/codegen/test_reduce_opt_validate.py
```

Expected: all tests pass with the pinned NPU-IR. Existing reduce, small-2D, and scalar fallback behavior is unchanged.

- [ ] **Step 5: Inspect local-only integration changes**

Run `git diff --check` in TileLang and inspect the sample repository status.
The strict test is already committed with Task 3. Do not force-add the excluded
MiniMax sample files to either repository.

---

### Task 5: Final Branch Verification and Documentation Status

**Files:**
- Verify: all files changed in Tasks 1-4
- Update locally: `/workspace/tilelang-samples/my_sample/minimax_m3_vl/TILE14_ALIGNED_1D_BRIDGE_DESIGN.md`

**Interfaces:**
- Consumes: all implementation commits.
- Produces: a clean TileLang feature branch, an updated local handoff document, and an explicit deferred-validation list.

- [ ] **Step 1: Run the relevant C++ build and tests**

Run from `/workspace/tilelang-samples`:

```bash
make build-tilelang
```

Run the available `tilelang_cpp_tests` filters covering TileView/fusion planner behavior. Record separately any known v0.1.0 target-registration or cost-model failures rather than changing unrelated expectations.

- [ ] **Step 2: Run the complete focused Python matrix**

Run the four files from Task 4 plus the local MiniMax test. Report exact pass/fail counts and retain the first failing artifact if any stage breaks.

- [ ] **Step 3: Inspect branch boundaries**

```bash
git status --short --branch
git diff --check
git log --oneline --decorate -8
```

Expected: TileLang contains only the planned source/tests/plan changes. The sample repo shows the TileLang gitlink plus excluded local MiniMax documents/tests; unrelated root changes remain untouched.

- [ ] **Step 4: Update the design document status**

Change only evidence-backed fields:

```text
文档状态 -> implementation complete only if all current-stage checks pass
已验证 -> exact commands and pass counts
尚未验证 -> latest NPU-IR, gem5 correctness, gem5 performance
```

Do not claim a performance improvement before the deferred gem5 comparison.

- [ ] **Step 5: Commit any final tracked cleanup**

If verification required a tracked test or code correction, first add a failing regression for it, then commit only those tracked files with a scoped message. The excluded local design and MiniMax files remain uncommitted by repository convention.
