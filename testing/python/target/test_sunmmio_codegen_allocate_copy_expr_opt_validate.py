import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout
from tilelang.utils.target import determine_target

from sunmmio_codegen_validation_utils import (
    assert_source_contains,
    validate_suvm_mlir_with_npuir_opt,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()

# Debug logs from this file:
import os

# print flag
os.environ["SUNMMIO_TEST_PRINT"] = "1"


def basic_allocate_copy_mma_kernel(
    M=32,
    N=32,
    K=32,
    block_M=32,
    block_N=32,
    block_K=32,
    dtype=T.float16,
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols
    grid_m = T.ceildiv(M, block_M)
    grid_n = T.ceildiv(N, block_N)

    shard_policy = T.MeshShardingPolicy(x=1, y=0)
    A_layout = make_zz_layout((M, K))
    B_layout = make_zz_layout((K, N))
    C_layout = make_zz_layout((M, N))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), shard_policy, device_mesh_config, dtype, layout=A_layout),  # type: ignore
        B: T.MeshTensor((K, N), shard_policy, device_mesh_config, dtype, layout=B_layout),  # type: ignore
        C: T.MeshTensor((M, N), shard_policy, device_mesh_config, dtype, layout=C_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            for by in T.serial(grid_m):
                for bx in T.serial(grid_n):
                    T.copy(A[by * block_M, 0], A_shared)
                    T.copy(B[0, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_shared, transpose_B=True)
                    T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def allocate_dma_copy_kernel_plus(
    M=512,
    N=512,
    K=256,
    block_M=32,
    block_N=32,
    block_K=32,
    dtype=T.float16,
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols
    grid_n = T.ceildiv(K, block_K)

    shard_policy = T.MeshShardingPolicy(x=1, y=0)
    A_layout = make_zz_layout((M, K))
    C_layout = make_zz_layout((M, N))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), shard_policy, device_mesh_config, dtype, layout=A_layout),  # type: ignore
        C: T.MeshTensor((M, N), shard_policy, device_mesh_config, dtype, layout=C_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            for bx in T.serial(grid_n):
                # Index exprs: add/mul/rem/min/max on DMA copy paths.
                ko = (bx + bx) % grid_n
                m_tile = bx * block_M
                n_tile = bx * block_N
                m_offset = m_tile + T.min(bx, 1)
                n_offset = n_tile + T.max(bx - 1, 0)

                T.copy(A[m_offset, ko * block_K], A_shared)
                T.copy(A[ko * block_K, n_offset], B_shared)
                T.gemm(A_shared, B_shared, C_shared, transpose_B=True)
                T.copy(C_shared, C[m_offset, n_offset])

    return main


def offset_region_copy_kernel_plus(
    M=512,
    N=512,
    block_M=64,
    block_N=64,
    tile_M=32,
    tile_N=32,
    dtype=T.float16,
):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols
    grid_m = T.ceildiv(M, block_M)
    grid_n = T.ceildiv(N, block_N)

    shard_policy = T.MeshShardingPolicy(x=1, y=0)
    A_layout = make_zz_layout((M, N))
    B_layout = make_zz_layout((M, N))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, N), shard_policy, device_mesh_config, dtype, layout=A_layout),  # type: ignore
        B: T.MeshTensor((M, N), shard_policy, device_mesh_config, dtype, layout=B_layout),  # type: ignore
    ):
        with T.Kernel(ncores) as _cid:
            A_rsram = T.alloc_shared((block_M, block_N), dtype, scope="shared.rsram")

            for by in T.serial(grid_m):
                for bx in T.serial(grid_n):
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
                        A[src_row : src_row + tile_M, src_col : src_col + tile_N],
                        A_rsram[rsram_row : rsram_row + tile_M, rsram_col : rsram_col + tile_N],
                    )
                    T.copy(
                        A_rsram[rsram_row : rsram_row + tile_M, rsram_col : rsram_col + tile_N],
                        B[src_row : src_row + tile_M, src_col : src_col + tile_N],
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
    div_mod_expr = tvm.tir.Mod(tvm.tir.Div(tvm.tir.Add(i, four), two), three)
    cond = tvm.tir.And(
        tvm.tir.LT(i, four),
        tvm.tir.Or(tvm.tir.GE(j, one), tvm.tir.Not(tvm.tir.EQ(i, j))),
    )
    select_expr = tvm.tir.Select(cond, tvm.tir.Add(i, j), tvm.tir.Sub(i, j))
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


def build_sunmmio_source_from_stmt(stmt):
    target = determine_target("Sunmmio", return_object=True)
    func = tvm.tir.PrimFunc([], stmt).with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
    mod = tvm.IRModule({"main": func})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(mod, target, "suvm").inspect_source()


def test_basic_allocate_copy_mma_codegen_validates_with_npuir_opt(tmp_path):
    validate_sunmmio_codegen_with_npuir_opt(
        basic_allocate_copy_mma_kernel(),
        tmp_path,
        mlir_filename="basic_allocate_copy_mma_suvm.mlir",
        expected_tokens=("suvm.alloc", "suvm.copy_async", "suvm.tc.mma"),
    )


def test_allocate_dma_copy_codegen_validates_with_npuir_opt(tmp_path):
    validate_sunmmio_codegen_with_npuir_opt(
        allocate_dma_copy_kernel_plus(),
        tmp_path,
        mlir_filename="allocate_dma_copy_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.tc.mma"),
    )


def test_offset_region_copy_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        offset_region_copy_kernel_plus(),
        tmp_path,
        mlir_filename="offset_region_copy_suvm.mlir",
        expected_tokens=("suvm.copy_async", "suvm.get_partitioned_tile_view"),
    )
    assert_source_contains(
        src,
        (
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
        ),
    )


def test_non_tile_expr_codegen_validates_with_npuir_opt(tmp_path):
    src = build_sunmmio_source_from_stmt(make_non_tile_expr_stmt())
    assert_source_contains(
        src,
        (
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
        ),
    )
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename="non_tile_expr_suvm.mlir",
    )


if __name__ == "__main__":
    tilelang.testing.main()
