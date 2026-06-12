import os

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
import tilelang.testing

from compile_pipeline import compile_test, target
from sunmmio_codegen_validation_utils import print_sunmmio_codegen_debug
from tilelang.utils.target import SUNMMIO_TARGET_DESC


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")

CODEGEN_BACKEND = "suvm"
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


def _to_device_kernel_func(func):
    return func.with_attr("global_symbol", "main").with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))


def codegen_sunmmio_suvm_from_kernel(kernel):
    mod = tvm.IRModule({"main": kernel})
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    _, device_mod = compile_test(mod, target=target, remove_header=True)
    target = determine_target("Sunmmio", return_object=True)
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    src = builder(device_mod, target, CODEGEN_BACKEND).inspect_source()
    print_sunmmio_codegen_debug(label="Lowered device", ir_obj=device_mod, mlir_src=src)
    return src


def build_sunmmio_source_from_stmt(stmt):
    target = determine_target("Sunmmio", return_object=True)
    func = _to_device_kernel_func(tvm.tir.PrimFunc([], stmt))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    src = builder(mod, target, CODEGEN_BACKEND).inspect_source()
    print_sunmmio_codegen_debug(label="Direct non-tile expr", ir_obj=mod, mlir_src=src)
    return src


def assert_contains_all(src: str, tokens):
    missing = [token for token in tokens if token not in src]
    assert not missing, f"missing expected tokens: {missing}\n{src}"


@target("Sunmmio")
def allocate_dma_copy_kernel(
    M=512,
    N=512,
    K=256,
    block_M=32,
    block_N=32,
    block_K=32,
    grid_M=8,
    grid_N=8,
    dtype=T.float16,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_N) as bx:
            # with T.Kernel(ncores) as _cid:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            # Index exprs: add/mul/rem/min/max on DMA copy paths.
            ko = (bx + bx) % T.ceildiv(K, block_K)
            m_tile = bx * block_M
            n_tile = bx * block_N
            m_offset = m_tile + T.min(bx, 1)
            n_offset = n_tile + T.max(bx - 1, 0)

            T.copy(A[m_offset, ko * block_K], A_shared)
            T.copy(A[ko * block_K, n_offset], B_shared)
            T.gemm(A_shared, B_shared, C_shared)
            T.copy(C_shared, C[m_offset, n_offset])

    return main


@target("Sunmmio")
def offset_region_copy_kernel(
    M=512,
    N=512,
    block_M=64,
    block_N=64,
    tile_M=32,
    tile_N=32,
    grid_M=4,
    grid_N=4,
    dtype=T.float16,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            grid_N,
            grid_M,
            threads=128,
        ) as (bx, by):
            A_rsram = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")

            # Integer exprs: add/sub/mul/div/rem for copy offsets.
            sum_idx = bx + by
            diff_idx = bx - by
            scaled_idx = sum_idx * 3 + 1
            div_idx = scaled_idx // 2
            mod_idx = div_idx % 7

            # Bool exprs: cmp/and/or/not feeding selects.
            lt_cmp = bx < by + 2
            le_cmp = by <= bx + 1
            ne_cmp = bx != by
            eq_cmp = bx == by
            gt_cmp = bx > by - 1
            ge_cmp = by >= bx - 1
            not_lt_le = T.Not(lt_cmp and le_cmp)
            row_cond = (lt_cmp and le_cmp) or (ne_cmp and not_lt_le)
            col_cond = gt_cmp or ge_cmp

            # Region exprs: select/min/max in tile-view indices.
            row_delta = T.Select(
                row_cond,
                T.Select(eq_cmp, T.min(mod_idx, 7), T.min(mod_idx + 1, 7)),
                T.max(diff_idx, 0),
            )
            col_delta = T.Select(
                col_cond,
                T.max((sum_idx * 5) % 11, 0),
                T.min(by + 3, 7),
            )
            src_row = by * block_M + 8 + row_delta
            src_col = bx * block_N + 8 + col_delta
            rsram_row = 8 + T.min(row_delta, 7)
            rsram_col = 8 + T.min(col_delta, 7)

            T.copy(
                A[
                    src_row : src_row + tile_M,
                    src_col : src_col + tile_N,
                ],
                A_rsram[rsram_row : rsram_row + tile_M, rsram_col : rsram_col + tile_N],
            )
            T.copy(
                A_rsram[rsram_row : rsram_row + tile_M, rsram_col : rsram_col + tile_N],
                B[
                    src_row : src_row + tile_M,
                    src_col : src_col + tile_N,
                ],
            )

    return main


def make_non_tile_expr_stmt():
    i = tvm.tir.Var("i", "int32")
    j = tvm.tir.Var("j", "int32")
    v = tvm.tir.Var("v", "int32")

    one = tvm.tir.IntImm("int32", 1)
    two = tvm.tir.IntImm("int32", 2)
    three = tvm.tir.IntImm("int32", 3)
    four = tvm.tir.IntImm("int32", 4)

    # Direct TIR keeps expr nodes that kernel lowering may canonicalize.
    int_expr = tvm.tir.Max(
        tvm.tir.Min(
            tvm.tir.FloorMod(
                tvm.tir.FloorDiv(
                    tvm.tir.Sub(tvm.tir.Mul(tvm.tir.Add(i, j), three), one),
                    two,
                ),
                four,
            ),
            tvm.tir.Add(i, four),
        ),
        j,
    )
    # Integer div/mod, bool/select, float arithmetic, and let coverage.
    div_mod_expr = tvm.tir.Mod(tvm.tir.Div(tvm.tir.Add(i, four), two), three)
    cond = tvm.tir.And(
        tvm.tir.LT(i, four),
        tvm.tir.Or(tvm.tir.GE(j, one), tvm.tir.Not(tvm.tir.EQ(i, j))),
    )
    select_expr = tvm.tir.Select(
        cond,
        tvm.tir.Add(i, j),
        tvm.tir.Sub(i, j),
    )
    float_expr = tvm.tir.Div(
        tvm.tir.Mul(
            tvm.tir.Add(
                tvm.tir.Cast("float32", tvm.tir.IntImm("int32", 7)),
                tvm.tir.FloatImm("float32", 1.5),
            ),
            tvm.tir.FloatImm("float32", 2.0),
        ),
        tvm.tir.FloatImm("float32", 3.0),
    )
    let_expr = tvm.tir.Let(
        v,
        tvm.tir.Add(i, j),
        tvm.tir.Max(v, tvm.tir.IntImm("int32", 0)),
    )
    let_stmt = tvm.tir.LetStmt(
        v,
        tvm.tir.Sub(i, j),
        tvm.tir.Evaluate(tvm.tir.Min(v, tvm.tir.IntImm("int32", 8))),
    )

    then_stmt = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(int_expr),
            tvm.tir.Evaluate(select_expr),
            tvm.tir.Evaluate(float_expr),
            tvm.tir.Evaluate(let_expr),
            let_stmt,
        ]
    )
    else_stmt = tvm.tir.Evaluate(div_mod_expr)
    inner = tvm.tir.IfThenElse(cond, then_stmt, else_stmt)
    inner_loop = tvm.tir.For(j, 0, 4, tvm.tir.ForKind.SERIAL, inner)
    return tvm.tir.For(i, 0, 8, tvm.tir.ForKind.SERIAL, inner_loop)


def test_sunmmio_codegen_allocate_and_dma_copy_paths():
    src = codegen_sunmmio_suvm_from_kernel(allocate_dma_copy_kernel())

    assert src.count("suvm.alloc") >= 3
    assert src.count("suvm.copy_async") >= 3
    assert src.count("suvm.get_partitioned_tile_view") >= 6
    assert "suvm.tc.mma" in src
    assert_contains_all(
        src,
        ["asram", "wsram", "rsram", "arith.addi", "arith.muli", "arith.remsi"],
    )


def test_sunmmio_codegen_offset_region_copy_paths():
    src = codegen_sunmmio_suvm_from_kernel(offset_region_copy_kernel())

    assert src.count("suvm.alloc") >= 1
    assert src.count("suvm.copy_async") >= 2
    assert src.count("suvm.get_partitioned_tile_view") >= 4
    assert_contains_all(
        src,
        [
            "arith.index_cast",
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.divsi",
            "arith.remsi",
            "arith.cmpi eq",
            "arith.cmpi ne",
            "arith.cmpi slt",
            "arith.cmpi sle",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "arith.select",
            "arith.minsi",
            "arith.maxsi",
        ],
    )


def test_sunmmio_codegen_non_tile_expr_ops():
    src = build_sunmmio_source_from_stmt(make_non_tile_expr_stmt())
    assert_contains_all(
        src,
        [
            "scf.for",
            "scf.if",
            "arith.index_cast",
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.divsi",
            "arith.remsi",
            "arith.cmpi",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "arith.select",
            "arith.sitofp",
            "arith.addf",
            "arith.mulf",
            "arith.divf",
            "arith.minsi",
            "arith.maxsi",
        ],
    )


if __name__ == "__main__":
    tilelang.testing.main()
    # test_sunmmio_codegen_allocate_and_dma_copy_paths()
