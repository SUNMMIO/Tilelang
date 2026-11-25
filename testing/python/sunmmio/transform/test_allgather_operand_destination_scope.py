import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.language.utils import region
from tilelang.layout import make_mxzz_layout, make_mxznz_layout
from tilelang.utils.target import determine_target

from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt


def make_gather_gemm(operand, explicit, reuse):
    """Collect one complete A/B panel per core on the test's 4x4 mesh."""
    mx = "custom[mxfp8]8"
    a_shape = (32, 128) if operand == "A" else (32, 512)
    w_shape = (128, 512) if operand == "B" else (512, 512)
    a_scope = "shared.asram" if explicit and operand == "A" else "shared"
    w_scope = "shared.wsram" if explicit and operand == "B" else "shared"

    @T.prim_func
    def main(
        A: T.MeshTensor(a_shape, T.placement.replicated(), mx, layout=make_mxzz_layout(a_shape, dtype=mx)),
        W: T.MeshTensor(w_shape, T.placement.replicated(), mx, layout=make_mxznz_layout(w_shape, dtype=mx)),
        Y: T.MeshTensor((32, 512), T.placement.replicated(), "float32"),
    ):
        with T.Kernel(T.mesh_ncores()):
            a = T.alloc_buffer((32, 512), mx, scope=a_scope)
            w = T.alloc_buffer((512, 512), mx, scope=w_scope)
            c = T.alloc_buffer((32, 512), "float32", scope="shared")
            if operand == "A":
                T.comm.all_gather(A, a, direction="horizontal", axis=1)
                T.copy(W, w)
            else:
                T.copy(A, a)
                T.comm.all_gather(W, w, direction="vertical", axis=0)
            T.gemm(a, w, c, clear_accum=True)
            if reuse:
                # The first GEMM has already assigned the destination scope
                # when InferSramScope encounters this second gather.
                if operand == "A":
                    T.comm.all_gather(A, a, direction="horizontal", axis=1)
                else:
                    T.comm.all_gather(W, w, direction="vertical", axis=0)
                T.gemm(a, w, c, clear_accum=True)
            T.copy(c, Y)

    return main


@pytest.mark.parametrize("operand", ["A", "B"])
@pytest.mark.parametrize("explicit", [False, True], ids=["inferred", "explicit"])
@pytest.mark.parametrize("reuse", [False, True], ids=["first_write", "reuse"])
def test_allgather_destination_may_be_operand_sram(tmp_path, operand, explicit, reuse):
    target = determine_target("sunmmio", return_object=True)
    with tvm.target.Target(target):
        kernel = make_gather_gemm(operand, explicit, reuse)
    source = validate_sunmmio_codegen_with_npuir_opt(kernel, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
    assert "suvm.mcast" in source
    assert "suvm.tc.mma" in source


def infer_gather_scope(src_scope="global", dst_scope="shared", address_scope=None, address_part="min"):
    """Pass-level input: a write descriptor may itself contain a real read."""
    tir = tvm.tir
    src = tir.decl_buffer((128, 32), "bfloat16", name="src", scope=src_scope)
    dst = tir.decl_buffer((512, 32), "bfloat16", name="dst", scope=dst_scope)
    start, extent = 0, 512
    if address_scope is not None:
        offsets = tir.decl_buffer((1,), "int32", name="offsets", scope=address_scope)
        if address_part == "min":
            start = offsets[0]
        else:
            extent = offsets[0]
    call = tir.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.tileop.comm_allgather"),
        region(src[0, 0], "r", 128, 32),
        region(dst[start, 0], "w", extent, 32),
        1,
        0,
        0,
        0,
    )
    target = determine_target("sunmmio", return_object=True)
    func = tir.PrimFunc([], tir.Evaluate(call)).with_attr("target", target)
    return tilelang.transform.InferSramScope()(tvm.IRModule({"main": func}))["main"]


def test_allgather_without_gemm_defaults_to_rsram():
    func = infer_gather_scope()
    destination = func.body.value.args[1].args[0].buffer
    assert destination.scope() == "shared.rsram"


@pytest.mark.parametrize("scope", ["shared.asram", "shared.wsram"])
def test_real_operand_sram_source_still_rejected(scope):
    with pytest.raises(tvm.error.InternalError, match="Invalid scope " + scope):
        infer_gather_scope(src_scope=scope)


@pytest.mark.parametrize("address_part", ["min", "extent"])
def test_destination_address_reads_still_checked(address_part):
    # The WSRAM destination is writable; an ordinary RSRAM index read is legal.
    infer_gather_scope(dst_scope="shared.wsram", address_scope="shared.rsram", address_part=address_part)
    # Skipping the descriptor's outer BufferLoad must not skip its index reads.
    with pytest.raises(tvm.error.InternalError, match="Invalid scope shared.asram"):
        infer_gather_scope(dst_scope="shared.wsram", address_scope="shared.asram", address_part=address_part)
