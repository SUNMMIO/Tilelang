import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.lower import lower
from testing.python.sunmmio.common.compile_pipeline import target


@target("sunmmio")
def invalid_view_kernel(case):
    @T.prim_func
    def main():
        with T.Kernel():
            mx = T.alloc_shared((32, 64), T.mxfp8, scope="shared.rsram")
            data, scale = T.mx_view(mx)
            if case == "duplicate":
                data2, scale2 = T.mx_view(mx)
                T.copy(data, data2)
            elif case == "scalar_read":
                T.evaluate(T.Cast(T.bfloat16, data[0, 0]))
            elif case == "scalar_write":
                scale[0, 0] = T.Cast(T.float8_e8m0fnu, T.bfloat16(1))
            elif case == "cast_copy":
                other = T.alloc_shared((32, 64), T.bfloat16, scope="shared.rsram")
                T.copy(other, data)
            elif case == "shape":
                other = T.alloc_shared((7, 32), T.float8_e8m0fnu, scope="shared.rsram")
                T.copy(other, scale)
            elif case == "pointer":
                T.evaluate(T.call_extern("handle", "escape", data.data))

    return main


@pytest.mark.parametrize(
    "case, message",
    [
        ("duplicate", "one declaration"),
        ("scalar_read", "standalone scalar reads"),
        ("scalar_write", "standalone scalar writes"),
        ("cast_copy", "matching dtypes"),
        ("shape", "(?i)(equal|mismatch)"),
        ("pointer", "raw pointers"),
    ],
)
def test_mx_view_invalid_use(case, message):
    with pytest.raises((ValueError, tvm.error.InternalError), match=message):
        lower(invalid_view_kernel(case), target="sunmmio")


@pytest.mark.parametrize(
    "shape,dtype,scope,message",
    [
        ((32, 32), T.bfloat16, "shared.rsram", "MXFP8 or MXFP4"),
        ((2, 32, 32), T.mxfp8, "shared.rsram", "rank-2"),
        ((32, 32), T.mxfp8, "shared.asram", "shared.rsram"),
    ],
)
def test_mx_view_invalid_input(shape, dtype, scope, message):
    parent = tvm.tir.decl_buffer(shape, dtype, scope=scope)
    with pytest.raises(ValueError, match=message):
        T.mx_view(parent)


@target("sunmmio")
def field_pipeline_kernel(touch_view=True, stages=2):
    @T.prim_func
    def main():
        with T.Kernel():
            mx = T.alloc_shared((32, 64), T.mxfp8, scope="shared.rsram")
            data, scale = T.mx_view(mx)
            peer = T.alloc_shared((32, 64), T.float8_e4m3fn, scope="shared.rsram")
            if touch_view:
                for _k in T.Pipelined(4, num_stages=stages):
                    T.copy(peer, data)
            else:
                T.copy(peer, data)
                independent_src = T.alloc_shared((32, 32), T.bfloat16, scope="shared.rsram")
                independent_a = T.alloc_shared((32, 32), T.bfloat16, scope="shared.asram")
                independent_b = T.alloc_shared((32, 32), T.bfloat16, scope="shared.wsram")
                independent_dst = T.alloc_shared((32, 32), T.bfloat16, scope="shared.rsram")
                for _k in T.Pipelined(4, num_stages=stages):
                    T.copy(independent_src, independent_a)
                    T.gemm(independent_a, independent_b, independent_dst, clear_accum=True)

    return main


@pytest.mark.parametrize("mode", ["greedy", "ilp"])
def test_mx_view_pipeline_rejected_with_declaration_outside_loop(mode):
    with (
        tilelang.transform.PassContext(config={"tl.sunmmio_pipeline_mode": mode}),
        pytest.raises(tvm.error.InternalError, match="automatic pipelining"),
    ):
        lower(field_pipeline_kernel(), target="sunmmio")


@pytest.mark.parametrize("config_key", ["tl.Simplify", "tir.Simplify"])
def test_mx_view_unsafe_buffer_propagation_rejected(config_key):
    with (
        tilelang.transform.PassContext(config={config_key: {"propagate_knowns_to_simplify_expressions": True}}),
        pytest.raises((ValueError, tvm.error.InternalError), match="buffer value propagation"),
    ):
        lower(field_pipeline_kernel(), target="sunmmio")


@pytest.mark.parametrize("mode", ["greedy", "ilp"])
def test_mx_view_unrelated_pipeline_remains_available(mode):
    with tilelang.transform.PassContext(config={"tl.sunmmio_pipeline_mode": mode}):
        lower(field_pipeline_kernel(touch_view=False), target="sunmmio")


def test_mx_view_zero_stages_is_a_serial_loop():
    lower(field_pipeline_kernel(stages=0), target="sunmmio")


@target("sunmmio")
def field_annotation_kernel(bits, role, annotation, used=True, nested=False):
    from tilelang.layout import make_mxzz_layout, make_row_major, make_zz_layout

    mx_dtype = T.mxfp8 if bits == 8 else T.mxfp4
    data_dtype = T.float8_e4m3fn if bits == 8 else T.float4_e2m1fn
    shape = (4, 32) if role == "scale" else (64, 64)
    layout = (
        make_mxzz_layout(shape, dtype=mx_dtype) if role == "parent" else make_zz_layout(shape) if role == "peer" else make_row_major(shape)
    )

    def annotate(buffer):
        if annotation == "layout":
            return T.annotate_layout({buffer: layout})
        return T.annotate_tileview({buffer: ([1, 64] if role == "scale" else [32, 32], [-2, -1])})

    @T.prim_func
    def main():
        with T.Kernel():
            mx = T.alloc_shared((64, 64), mx_dtype, scope="shared.rsram")
            data, scale = T.mx_view(mx)
            peer = T.alloc_shared((64, 64), data_dtype, scope="shared.rsram")
            alias = scale if role == "scale" else data
            if role == "parent":
                alias = mx
            if role == "peer":
                alias = peer
            if nested:
                with T.block("annotation"):
                    annotate(alias)
            else:
                annotate(alias)
            if used:
                if role == "scale":
                    scales = T.alloc_shared((4, 32), T.float8_e8m0fnu, scope="shared.rsram")
                    T.copy(scale, scales)
                else:
                    T.copy(data, peer)

    return main


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("role,annotation", [("data", "layout"), ("scale", "layout"), ("scale", "tileview")])
@pytest.mark.parametrize("used,nested", [(True, False), (False, False), (True, True)])
def test_mx_field_annotations_rejected_before_normalization(bits, role, annotation, used, nested):
    with pytest.raises(ValueError, match=f"T.annotate_{annotation}.*mx_view.*{role}"):
        lower(field_annotation_kernel(bits, role, annotation, used, nested), target="sunmmio")


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("role,annotation", [("parent", "layout"), ("peer", "layout"), ("data", "tileview"), ("peer", "tileview")])
def test_mx_parent_peer_and_data_tileview_annotations_remain_supported(bits, role, annotation):
    artifact = lower(field_annotation_kernel(bits, role, annotation), target="sunmmio")
    assert "suvm.unpack" in artifact.kernel_source
