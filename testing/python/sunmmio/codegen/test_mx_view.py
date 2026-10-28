"""MX field aliases through the production SunMMIO compiler."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.utils.target as target_utils
from tilelang.engine.lower import lower
from tilelang.layout import CuteLayout, get_mx_scale_shape, make_mx_row_major_layout, make_mxznz_layout, make_mxzz_layout
from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import validate_suvm_mlir_with_npuir_opt
from testing.python.sunmmio.common.mx_view_validation import assert_mx_view_resolved


def mx_layout(kind, shape, dtype):
    if kind == "mxznz" and shape[0] % 64:
        covered = ((shape[0] + 63) // 64 * 64, (shape[1] + 31) // 32 * 32)
        base = make_mxznz_layout(covered, dtype=dtype)

        def query(field):
            return tilelang.tvm.ffi.get_global_func("tl.CuteLayout_" + field)(base)

        return CuteLayout(shape, query("mode_shape"), query("mode_stride"), query("dim_levels"))._inner
    return {"mxzz": make_mxzz_layout, "mxznz": make_mxznz_layout, "row_major": make_mx_row_major_layout}[kind](shape, dtype=dtype)


@target("sunmmio")
def field_copy_kernel(bits=8, kind="mxzz", shape=(64, 128), annotate=True, only="both", scope="shared.rsram"):
    mx_dtype = T.mxfp8 if bits == 8 else T.mxfp4
    data_dtype = T.float8_e4m3fn if bits == 8 else T.float4_e2m1fn
    layout = mx_layout(kind, shape, mx_dtype)
    scale_shape = tuple(int(x) for x in get_mx_scale_shape(layout, mx_dtype))

    @T.prim_func
    def main():
        with T.Kernel():
            mx = T.alloc_shared(shape, mx_dtype, scope=scope)
            if annotate:
                T.annotate_layout({mx: layout})
            data, scale = T.mx_view(mx)
            if only != "scale":
                payload = T.alloc_shared(shape, data_dtype, scope="shared.rsram")
                T.copy(payload, data)
                T.copy(data, payload)
            if only != "data":
                scales = T.alloc_shared(scale_shape, T.float8_e8m0fnu, scope="shared.rsram")
                T.copy(scales, scale)
                T.copy(scale, scales)

    return main


@target("sunmmio")
def field_compute_kernel(bits=8, kind="mxzz", shape=(64, 128)):
    mx_dtype = T.mxfp8 if bits == 8 else T.mxfp4
    data_dtype = T.float8_e4m3fn if bits == 8 else T.float4_e2m1fn
    layout = mx_layout(kind, shape, mx_dtype)

    @T.prim_func
    def main():
        with T.Kernel():
            mx = T.alloc_shared(shape, mx_dtype, scope="shared.rsram")
            T.annotate_layout({mx: layout})
            data, scale = T.mx_view(mx)
            for i, j in T.Tiles(data.shape):
                data[i, j] = T.Cast(data_dtype, T.bfloat16(1))
            for i, j in T.Tiles(scale.shape):
                scale[i, j] = T.Cast(T.float8_e8m0fnu, T.bfloat16(1))
            output = T.alloc_shared(shape, T.bfloat16, scope="shared.rsram")
            for i, j in T.Tiles(shape):
                output[i, j] = T.Cast(T.bfloat16, data[i, j])

    return main


def compile_and_validate(kernel, tmp_path, configs=None):
    with tilelang.transform.PassContext(config=configs or {}):
        artifact = lower(kernel, target="sunmmio")
    source = artifact.kernel_source
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
    assert_mx_view_resolved(artifact.device_mod)
    assert source.count("suvm.unpack ") == 1
    return source


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("kind", ["mxzz", "mxznz", "row_major"])
@pytest.mark.parametrize("strict", [False, True])
def test_mx_view_field_copy(bits, kind, strict, tmp_path):
    previous = target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    try:
        target_utils.set_sunmmio_region_validation(strict)
        source = compile_and_validate(field_copy_kernel(bits, kind), tmp_path)
    finally:
        target_utils.set_sunmmio_region_validation(previous)
    # One MX plus the two explicit copy peers; the fields own no storage.
    assert source.count("suvm.alloc()") == 3


def test_mx_view_inferred_rsram_scope(tmp_path):
    compile_and_validate(field_copy_kernel(scope="shared.dyn", annotate=False), tmp_path)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("kind", ["mxzz", "mxznz", "row_major"])
def test_mx_view_direct_computation(bits, kind, tmp_path):
    source = compile_and_validate(field_compute_kernel(bits, kind), tmp_path)
    assert source.count("suvm.alloc()") == 2


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("operation", ["quant_dequant", "gemm"])
def test_mx_view_application_kernels_validate_without_simulator(bits, operation, tmp_path):
    from testing.python.sunmmio.common.mx_view_kernels import mx_view_gemm_numeric_kernel, mx_view_quant_numeric_kernel

    kernel = mx_view_quant_numeric_kernel(bits, use_view=True) if operation == "quant_dequant" else mx_view_gemm_numeric_kernel(bits)
    artifact = lower(kernel, target="sunmmio")
    assert_mx_view_resolved(artifact.device_mod)
    assert tilelang.tvm.tir.analysis.verify_well_formed(artifact.device_mod)
    assert "suvm.unpack" in artifact.kernel_source
    validate_suvm_mlir_with_npuir_opt(artifact.kernel_source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))


@pytest.mark.parametrize("only", ["data", "scale"])
@pytest.mark.parametrize("force_inline", [False, True])
def test_mx_view_single_field_preserves_parent(only, force_inline, tmp_path):
    source = compile_and_validate(
        field_copy_kernel(only=only, annotate=False),
        tmp_path,
        {"tl.force_let_inline": force_inline},
    )
    assert source.count("suvm.alloc()") == 2


@target("sunmmio")
def two_mx_kernel(bits=8, kind="mxzz", unroll=False):
    mx_dtype = T.mxfp8 if bits == 8 else T.mxfp4
    layout = mx_layout(kind, (64, 64), mx_dtype)

    @T.prim_func
    def main():
        with T.Kernel():
            for _repeat in T.unroll(2 if unroll else 1):
                first = T.alloc_shared((64, 64), mx_dtype, scope="shared.rsram")
                second = T.alloc_shared((64, 64), mx_dtype, scope="shared.rsram")
                T.annotate_layout({first: layout, second: layout})
                a, sa = T.mx_view(first)
                b, sb = T.mx_view(second)
                T.copy(a, b)
                T.copy(sa, sb)

    return main


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("kind", ["mxzz", "mxznz", "row_major"])
@pytest.mark.parametrize("unroll", [False, True])
def test_mx_view_cross_parent_copy_and_unroll(bits, kind, unroll, tmp_path):
    artifact = lower(two_mx_kernel(bits, kind, unroll), target="sunmmio")
    source = artifact.kernel_source
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
    # Pure field handles may be hoisted or shared across expanded iterations.
    assert 2 <= source.count("suvm.unpack ") <= (4 if unroll else 2)
    assert_mx_view_resolved(artifact.device_mod)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize(
    "kind,shape",
    [
        ("mxzz", (49, 64)),
        ("mxznz", (49, 64)),
        ("row_major", (17, 96)),
        ("row_major", (32, 32)),
        ("mxzz", (1, 64)),
        ("mxzz", (64, 1)),
    ],
)
def test_mx_view_padded_logical_shape(bits, kind, shape, tmp_path):
    source = compile_and_validate(field_compute_kernel(bits, kind, shape), tmp_path)
    assert source.count("suvm.alloc()") == 2


@pytest.mark.parametrize("bits", [8, 4])
def test_mx_view_quant_dequant_removes_intermediate_storage(bits, tmp_path):
    from testing.python.sunmmio.codegen.test_mx_quant_dequant_opt_validate import mx_ocp_quant_dequant_full_chain_kernel_for_debug

    args = (T.mxfp8, T.float8_e4m3fn, "float8_e4m3fn", 448.0) if bits == 8 else (T.mxfp4, T.float4_e2m1fn, "float4_e2m1fn", 6.0)
    sources = []
    for use_view in [False, True]:
        kernel = mx_ocp_quant_dequant_full_chain_kernel_for_debug(*args, use_view=use_view)
        source = lower(kernel, target="sunmmio").kernel_source
        validate_suvm_mlir_with_npuir_opt(
            source,
            tmp_path,
            mlir_filename=f"quant_{use_view}.mlir",
            opt_args=("--verify-each", "--suvm-to-llvm-pipeline"),
        )
        sources.append(source)
    assert sources[0].count("suvm.alloc()") - sources[1].count("suvm.alloc()") == 4
    assert sources[0].count("suvm.unpack ") == 2
    assert sources[1].count("suvm.unpack ") == 1


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("k", [32, 96, 1056])
@pytest.mark.parametrize("operation", ["pack", "unpack", "roundtrip"])
def test_mx_view_numeric_kernel_lowering(bits, k, operation, tmp_path):
    from testing.python.sunmmio.common.mx_view_kernels import mx_row_major_numeric_kernel

    compile_and_validate(mx_row_major_numeric_kernel(k, bits, operation, use_view=True), tmp_path)


@target("sunmmio")
def parent_transfer_kernel():
    shape = (128, 32)
    layout = make_mxzz_layout(shape, dtype=T.mxfp8)

    @T.prim_func
    def main(
        Input: T.MeshTensor(shape, T.placement.replicated(), T.mxfp8, layout=layout),
        Output: T.MeshTensor(shape, T.placement.replicated(), T.mxfp8, layout=layout),
    ):
        with T.Kernel():
            mx = T.alloc_shared(shape, T.mxfp8, scope="shared.rsram")
            peer = T.alloc_shared(shape, T.float8_e4m3fn, scope="shared.rsram")
            data, scale = T.mx_view(mx)
            T.copy(Input, mx)
            T.copy(data, peer)
            T.copy(peer, data)
            T.copy(mx, Output)
            T.copy(Input, mx)
            T.copy(data, peer)

    return main


def test_mx_view_parent_dma_synchronization(tmp_path):
    source = compile_and_validate(parent_transfer_kernel(), tmp_path)
    first_load = source.index("suvm.tile.load")
    assert source.index("suvm.sync odma") < first_load
    first_store = source.index("suvm.tile.store")
    assert source.index("suvm.sync vector,rsram", first_store) < source.rfind("suvm.copy_async")


@target("sunmmio")
def inferred_weight_layout_kernel(bits):
    mx_dtype = T.mxfp8 if bits == 8 else T.mxfp4
    data_dtype = T.float8_e4m3fn if bits == 8 else T.float4_e2m1fn

    @T.prim_func
    def main():
        with T.Kernel():
            activation = T.alloc_shared((64, 64), mx_dtype, scope="shared.asram")
            weight_parent = T.alloc_shared((64, 64), mx_dtype, scope="shared.rsram")
            weight = T.alloc_shared((64, 64), mx_dtype, scope="shared.wsram")
            output = T.alloc_shared((64, 64), T.bfloat16, scope="shared.rsram")
            data, scale = T.mx_view(weight_parent)
            for i, j in T.Tiles(data.shape):
                data[i, j] = T.Cast(data_dtype, T.bfloat16(1))
            for i, j in T.Tiles(scale.shape):
                scale[i, j] = T.Cast(T.float8_e8m0fnu, T.bfloat16(1))
            T.copy(weight_parent, weight)
            T.clear(output)
            T.gemm(activation, weight, output)

    return main


@pytest.mark.parametrize("bits", [8, 4])
def test_mx_view_parent_layout_follows_gemm_consumer(bits, tmp_path):
    artifact = lower(inferred_weight_layout_kernel(bits), target="sunmmio")
    expected = make_mxznz_layout((64, 64), dtype=T.mxfp8 if bits == 8 else T.mxfp4)
    parents = []
    for func in artifact.device_mod.functions.values():
        for buffer, layout in func.attrs["layout_map"].items():
            if buffer.name == "weight_parent":
                parents.append(layout)
    assert parents and all(tilelang.tvm.ffi.get_global_func("tl.IsSameLayout")(layout, expected) for layout in parents)
    validate_suvm_mlir_with_npuir_opt(artifact.kernel_source, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))


@target("sunmmio")
def legacy_view_kernel():
    # Mesh symbols are resolved by the production pipeline before verification.
    @tilelang.tvm.script.tir.prim_func(check_well_formed=False)
    def main():
        for _cid in T.thread_binding(T.mesh_ncores(), thread="blockIdx.x"):
            with T.block("tilelang_root"):
                mx = T.alloc_shared((64, 64), T.mxfp8, scope="shared.rsram")
                data, scale = T.mx_view(mx)
                peer = T.alloc_shared((64, 64), T.float8_e4m3fn, scope="shared.rsram")
                T.copy(peer, data)

    return main.with_attr("tl.sunmmio.mesh_nrows_var", T.mesh_nrows()).with_attr("tl.sunmmio.mesh_ncols_var", T.mesh_ncols())


def test_mx_view_tvm_legacy_parser(tmp_path):
    kernel = legacy_view_kernel()
    artifact = lower(kernel, target="sunmmio")
    assert tilelang.tvm.tir.analysis.verify_well_formed(artifact.device_mod)
    for func in artifact.device_mod.functions.values():
        assert tilelang.tvm.tir.analysis.verify_ssa(func)
    compile_and_validate(kernel, tmp_path)


@target("sunmmio")
def field_shape_alias_kernel():
    @T.prim_func
    def main():
        with T.Kernel():
            mx = T.alloc_shared((64, 64), T.mxfp8, scope="shared.rsram")
            data, scale = T.mx_view(mx)
            rows, cols = scale.shape
            rows_alias = rows + 0
            cols_alias = cols // 2 * 2
            peer = T.alloc_shared((rows_alias, cols_alias), T.float8_e8m0fnu, scope="shared.rsram")
            T.copy(peer, scale)
            if rows_alias == 0:
                bad = T.alloc_shared((7, 32), T.float8_e8m0fnu, scope="shared.rsram")
                T.copy(bad, scale)
            for _empty in T.serial(rows_alias - 4):
                bad_loop = T.alloc_shared((7, 32), T.float8_e8m0fnu, scope="shared.rsram")
                T.copy(bad_loop, scale)
            for i, j in T.Tiles([rows_alias, cols_alias]):
                scale[i, j] = T.Cast(T.float8_e8m0fnu, T.bfloat16(1))
            T.copy(scale, peer)

    return main


@pytest.mark.parametrize("force_inline", [False, True])
def test_mx_view_shape_let_aliases_resolve_in_metadata(force_inline, tmp_path):
    source = compile_and_validate(field_shape_alias_kernel(), tmp_path, {"tl.force_let_inline": force_inline})
    assert source.count("suvm.alloc()") == 2


@target("sunmmio")
def conditional_view_kernel():
    @T.prim_func
    def main():
        with T.Kernel() as cid:
            mx = T.alloc_shared((64, 64), T.mxfp8, scope="shared.rsram")
            peer = T.alloc_shared((64, 64), T.float8_e4m3fn, scope="shared.rsram")
            if cid % 2 == 0:
                data, scale = T.mx_view(mx)
                for _iteration in T.serial(2):
                    T.copy(peer, data)
                T.copy(data, peer)

    return main


def test_mx_view_conditional_declaration_and_serial_reuse(tmp_path):
    compile_and_validate(conditional_view_kernel(), tmp_path)


@target("sunmmio")
def sliced_fields_kernel(bits, kind):
    mx_dtype = T.mxfp8 if bits == 8 else T.mxfp4
    data_dtype = T.float8_e4m3fn if bits == 8 else T.float4_e2m1fn
    layout = mx_layout(kind, (64, 128), mx_dtype)

    @T.prim_func
    def main():
        with T.Kernel():
            mx = T.alloc_shared((64, 128), mx_dtype, scope="shared.rsram")
            T.annotate_layout({mx: layout})
            data, scale = T.mx_view(mx)
            peer = T.alloc_shared((32, 32), data_dtype, scope="shared.rsram")
            scale_peer = T.alloc_shared((1, scale.shape[1]), T.float8_e8m0fnu, scope="shared.rsram")
            for block in T.serial(2):
                T.copy(data[block * 32 : block * 32 + 32, 32:64], peer)
                T.copy(peer, data[block * 32 : block * 32 + 32, 32:64])
            for row in T.serial(scale.shape[0]):
                T.copy(scale[row : row + 1, :], scale_peer)
                T.copy(scale_peer, scale[row : row + 1, :])

    return main


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("kind", ["mxzz", "mxznz", "row_major"])
def test_mx_view_aligned_slices_with_dynamic_offsets(bits, kind, tmp_path):
    compile_and_validate(sliced_fields_kernel(bits, kind), tmp_path)


@target("sunmmio")
def deferred_region_kernel(case, reverse=False):
    @T.prim_func
    def main():
        with T.Kernel() as cid:
            mx = T.alloc_shared((64, 64), T.mxfp8, scope="shared.rsram")
            data, scale = T.mx_view(mx)
            if case == "auto_shape":
                peer = T.alloc_shared(scale.shape, T.float8_e8m0fnu)
                if reverse:
                    T.copy(peer, scale)
                else:
                    T.copy(scale, peer)
            elif case == "auto_static":
                peer = T.alloc_shared((4, 32), T.float8_e8m0fnu)
                if reverse:
                    T.copy(peer, scale)
                else:
                    T.copy(scale, peer)
            else:
                peer = T.alloc_shared((1, 32), T.float8_e8m0fnu, scope="shared.rsram")
                row = cid % 4
                alias = row + 1
                index = alias - 1
                if case == "let":
                    if reverse:
                        T.copy(peer, scale[index : index + 1, :])
                    else:
                        T.copy(scale[index : index + 1, :], peer)
                elif case == "load":
                    if reverse:
                        T.copy(peer, scale[index, 0])
                    else:
                        T.copy(scale[index, 0], peer)
                else:
                    for i in T.serial(4):
                        loop_index = i
                        if reverse:
                            T.copy(peer, scale[loop_index : loop_index + 1, :])
                        else:
                            T.copy(scale[loop_index : loop_index + 1, :], peer)

    return main


@pytest.mark.parametrize("case", ["auto_shape", "auto_static", "let", "load", "loop"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("force_inline", [False, True])
def test_mx_view_deferred_regions_preserve_bindings(case, reverse, strict, force_inline, tmp_path):
    previous = target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    try:
        target_utils.set_sunmmio_region_validation(strict)
        compile_and_validate(deferred_region_kernel(case, reverse), tmp_path, {"tl.force_let_inline": force_inline})
    finally:
        target_utils.set_sunmmio_region_validation(previous)


@pytest.mark.parametrize("case", ["negative_last", "negative_bounds", "negative_end", "both_negative", "loop"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("force_inline", [False, True])
def test_mx_view_negative_regions_compile(case, reverse, force_inline, tmp_path):
    from testing.python.sunmmio.common.mx_view_kernels import mx_view_region_numeric_kernel

    previous = target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    try:
        target_utils.set_sunmmio_region_validation(False)
        compile_and_validate(mx_view_region_numeric_kernel(case, reverse), tmp_path, {"tl.force_let_inline": force_inline})
    finally:
        target_utils.set_sunmmio_region_validation(previous)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("force_inline", [False, True])
def test_mx_view_negative_region_strict_diagnostic(reverse, force_inline):
    from testing.python.sunmmio.common.mx_view_kernels import mx_view_region_numeric_kernel

    previous = target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    try:
        target_utils.set_sunmmio_region_validation(True)
        with (
            tilelang.transform.PassContext(config={"tl.force_let_inline": force_inline}),
            pytest.raises(ValueError, match="starts outside buffer shape"),
        ):
            lower(mx_view_region_numeric_kernel("negative_last", reverse), target="sunmmio")
    finally:
        target_utils.set_sunmmio_region_validation(previous)
