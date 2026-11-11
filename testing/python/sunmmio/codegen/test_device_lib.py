"""Device library lowering, layout and synchronization regression coverage."""

import pytest
import tilelang.language as T
from tilelang.layout import make_aligned_row_major, make_zz_layout
from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt


@target("sunmmio")
def topk_kernel(rows=1, cols=512, dtype="float32", zz=False, dynamic=False, selection_k=8, stages=0, output_cols=512):
    placement = T.placement.replicated()
    shape = (rows, cols)
    output_shape = (rows, output_cols)
    input_layout = make_zz_layout(shape) if zz else make_aligned_row_major(shape, dtype, align_bytes=1024)
    value_layout = make_aligned_row_major(output_shape, dtype, align_bytes=1024)
    index_layout = make_aligned_row_major(output_shape, "int32", align_bytes=1024)

    @T.prim_func
    def main(
        A: T.MeshTensor(shape, placement, dtype, layout=input_layout),
        V: T.MeshTensor(output_shape, placement, dtype, layout=value_layout),
        I: T.MeshTensor(output_shape, placement, "int32", layout=index_layout),
        k: T.int32,
    ):
        with T.Kernel():
            x = T.alloc_shared(shape, dtype, scope="shared.rsram")
            values = T.alloc_shared(output_shape, dtype, scope="shared.rsram")
            indices = T.alloc_shared(output_shape, "int32", scope="shared.rsram")
            for _iteration in T.Pipelined(2 if stages else 1, num_stages=stages):
                T.copy(A, x)
                T.topk(x, values, indices, k if dynamic else selection_k)
                T.copy(values, V)
                T.copy(indices, I)

    return main


@target("sunmmio")
def unary_kernel(name="exp_refined", rows=1, inplace=False, zz=False, dtype="float32"):
    shape = (rows, 512)
    placement = T.placement.replicated()
    layout = make_zz_layout(shape) if zz else make_aligned_row_major(shape, dtype, align_bytes=1024)
    operation = getattr(T, name)

    @T.prim_func
    def main(A: T.MeshTensor(shape, placement, dtype, layout=layout), B: T.MeshTensor(shape, placement, dtype, layout=layout)):
        with T.Kernel():
            x = T.alloc_shared(shape, dtype, scope="shared.rsram")
            T.copy(A, x)
            if inplace:
                operation(x, x)
                T.copy(x, B)
            else:
                y = T.alloc_shared(shape, dtype, scope="shared.rsram")
                operation(x, y)
                T.copy(y, B)

    return main


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("rows,zz", [(1, False), (32, True)])
@pytest.mark.parametrize("dynamic", [False, True])
def test_topk_codegen(tmp_path, dtype, rows, zz, dynamic):
    source = validate_sunmmio_codegen_with_npuir_opt(
        topk_kernel(rows=rows, dtype=dtype, zz=zz, dynamic=dynamic),
        tmp_path,
        expected_tokens=("suvm.device_lib.topk", "suvm.sync"),
        opt_args=("--verify-each", "--suvm-device-validate"),
    )
    after = source.split("suvm.device_lib.topk", 1)[1]
    assert "suvm.sync vector, rsram" in after or "suvm.sync vector,rsram" in after


@pytest.mark.parametrize(
    "name,op",
    [("exp_refined", "exp_fast"), ("log_refined", "ln_fast"), ("reciprocal_refined", "recip_fast"), ("rsqrt_refined", "rsqrt_fast")],
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("rows,zz", [(1, False), (32, True)])
def test_unary_codegen(tmp_path, name, op, inplace, rows, zz):
    validate_sunmmio_codegen_with_npuir_opt(
        unary_kernel(name, rows, inplace, zz),
        tmp_path,
        expected_tokens=(f"suvm.device_lib.{op}", "suvm.sync"),
        opt_args=("--verify-each", "--suvm-device-validate"),
    )


@pytest.mark.parametrize(
    "dtype,k,message",
    [
        ("float16", 8, "does not support float16"),
        ("float32", 513, "output width must be at least k"),
        ("float32", -1, "k must be non-negative"),
    ],
)
def test_topk_invalid_contract(tmp_path, dtype, k, message):
    with pytest.raises(Exception, match=message):
        validate_sunmmio_codegen_with_npuir_opt(topk_kernel(dtype=dtype, selection_k=k), tmp_path)


def test_topk_pipeline(tmp_path):
    validate_sunmmio_codegen_with_npuir_opt(
        topk_kernel(stages=2),
        tmp_path,
        expected_tokens=("suvm.device_lib.topk", "suvm.sync"),
        opt_args=("--verify-each", "--suvm-device-validate"),
    )


def test_decode_topk_64000_to_2048(tmp_path):
    validate_sunmmio_codegen_with_npuir_opt(
        topk_kernel(cols=64000, selection_k=2048, output_cols=2048),
        tmp_path,
        expected_tokens=("suvm.device_lib.topk",),
        opt_args=("--verify-each", "--suvm-device-validate"),
    )


def test_refined_rejects_bfloat16(tmp_path, capfd):
    with pytest.raises(Exception, match="Invalid device library call: tl.device_lib.exp_fast"):
        validate_sunmmio_codegen_with_npuir_opt(unary_kernel(dtype="bfloat16"), tmp_path)
    assert "operand types ('bf16', 'bf16') match no dispatch table entry" in capfd.readouterr().err


@pytest.mark.parametrize("read_modified_input", [False, True])
def test_device_lib_sync_effects(read_modified_input):
    import tilelang
    from tilelang import tvm
    from tilelang.language.device_lib import _region
    from tilelang.utils.target import determine_target

    buffers = [
        tvm.tir.decl_buffer((1, 512), dtype, name=name, scope="shared.rsram")
        for name, dtype in [("x", "float32"), ("v", "float32"), ("i", "int32"), ("z", "float32")]
    ]
    x, v, indices, z = buffers
    topk = tvm.tir.Evaluate(T._device_lib_call("topk", _region(x, "rw"), _region(v, "w"), _region(indices, "w"), 8))
    if read_modified_input:
        # topk writes its input too: a following scalar read must wait.
        statements = [topk, tvm.tir.Evaluate(x[0, 0])]
    else:
        # Even disjoint operands share the library's vector/RSRAM units.
        independent = tvm.tir.Evaluate(T._device_lib_call("exp_fast", _region(z, "r"), _region(z, "w")))
        statements = [independent, topk]
    body = tvm.tir.SeqStmt(statements)
    for buffer in reversed(buffers):
        body = tvm.tir.DeclBuffer(buffer, body)
    body = tvm.tir.AttrStmt(tvm.te.thread_axis("blockIdx.x"), "thread_extent", 16, body)
    func = tvm.tir.PrimFunc([buffer.data for buffer in buffers], body).with_attr("global_symbol", "main")
    func = func.with_attr("tir.is_global_func", True)
    mod = tvm.tir.transform.BindTarget(determine_target("sunmmio", return_object=True))(tvm.IRModule({"main": func}))
    result = tilelang.transform.InjectSunmmioSync()(mod)
    events = []

    def collect(node):
        if not isinstance(node, tvm.tir.Evaluate):
            return
        value = node.value
        if isinstance(value, tvm.tir.BufferLoad):
            events.append("read")
        elif isinstance(value, tvm.tir.Call):
            if value.op == tvm.ir.Op.get("tl.device_lib"):
                events.append(value.args[0].value)
            elif value.op == tvm.ir.Op.get("tl.sunmmio_sync"):
                assert int(value.args[0]) == 96
                events.append("sync")

    tvm.tir.stmt_functor.post_order_visit(result["main"].body, collect)
    if read_modified_input:
        assert events == ["topk", "sync", "read"]
    else:
        assert events == ["exp_fast", "sync", "topk"]


@target("sunmmio")
def unary_region_kernel():
    placement = T.placement.replicated()
    input_layout = make_aligned_row_major((4, 512), "float32", align_bytes=1024)
    output_layout = make_aligned_row_major((2, 512), "float32", align_bytes=1024)

    @T.prim_func
    def main(
        A: T.MeshTensor((4, 512), placement, "float32", layout=input_layout),
        B: T.MeshTensor((2, 512), placement, "float32", layout=output_layout),
    ):
        with T.Kernel():
            x = T.alloc_shared((4, 512), "float32", scope="shared.rsram")
            y = T.alloc_shared((2, 512), "float32", scope="shared.rsram")
            T.copy(A, x)
            T.exp_refined(x[1:3, :], y)
            T.copy(y, B)

    return main


def test_unary_region_with_different_parent_shapes(tmp_path):
    validate_sunmmio_codegen_with_npuir_opt(
        unary_region_kernel(),
        tmp_path,
        expected_tokens=("suvm.device_lib.exp_fast",),
        opt_args=("--verify-each", "--suvm-device-validate"),
    )


def test_topk_tir_script_roundtrip():
    from tilelang import tvm

    @T.prim_func
    def call_topk(
        A: T.Tensor((1, 512), "float32", scope="shared.rsram"),
        V: T.Tensor((1, 512), "float32", scope="shared.rsram"),
        I: T.Tensor((1, 512), "int32", scope="shared.rsram"),
    ):
        T.topk(A, V, I, 8)

    from types import SimpleNamespace
    from tvm.script import tir as parser

    namespace = SimpleNamespace(
        **{
            **vars(parser),
            "region": lambda *args: tvm.tir.call_intrin("handle", tvm.ir.Op.get("tl.tileop.region"), *args),
            "_device_lib": T._device_lib,
        }
    )
    restored = tvm.script.from_source(call_topk.script(), extra_vars={"T": namespace})
    tvm.ir.assert_structural_equal(call_topk, restored, map_free_vars=True)
