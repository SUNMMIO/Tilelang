"""
  test_reduce_sunmmio.py 用法说明：

  默认运行 tiled3d reduce kernel：

      python3 mlir_codegen/test_reduce_sunmmio.py

  默认等价于：

      TL_REDUCE_CASE=tiled3d TL_REDUCE_AXIS=1 TL_REDUCE_CLEAR=0 \
        python3 mlir_codegen/test_reduce_sunmmio.py

  常用环境变量：

  1. TL_REDUCE_CASE=tiled3d

        跑默认 tiled 3D kernel。此时可用 TL_REDUCE_AXIS=0/1/2
        切换 reduce 轴。

  2. TL_REDUCE_CASE=0

        跑脚本里 REDUCE_TEST_CASES 的第 0 个 case。范围当前是 0..10。
        REDUCE_TEST_CASES 的格式为：

            (shape, reduce_axis, expected_in_tile_reduce, clear)

  3. TL_REDUCE_CLEAR=1 或 TL_REDUCE_CLEAR=0

        覆盖当前 case 的 clear 参数。可用值：
        1/0, true/false, yes/no, on/off。

  4. TL_REDUCE_DTYPE=float16

        设置输入和输出 dtype。默认 float16。

  5. TL_REDUCE_SKIP_CODEGEN=1

        只跑 TileLang pass pipeline，生成 TIR/AST，不跑最后 SunMMIO MLIR codegen。

  6. TL_REDUCE_LOG_ROOT=/tmp/reduce_logs

        自定义 log 根目录。最终 AST 和 MLIR 会保存到：

            ${TL_REDUCE_LOG_ROOT}/<case_label>/final_ast.txt
            ${TL_REDUCE_LOG_ROOT}/<case_label>/final_mlir.mlir

        case_label 会包含 reduce axis 和 clear，例如：

            tiled3d_axis1_clear_false
            shape0_axis0_1024_clear_true

  常用批量跑法：

  for axis in 0 1 2; do
    TL_REDUCE_AXIS=$axis python3 mlir_codegen/test_reduce_sunmmio.py
  done

  for case in 0 1 2 3 4 5 6 7 8 9 10; do
    TL_REDUCE_CASE=$case python3 mlir_codegen/test_reduce_sunmmio.py
  done

  for clear in 0 1; do
    TL_REDUCE_CASE=tiled3d TL_REDUCE_AXIS=1 TL_REDUCE_CLEAR=$clear \
      python3 mlir_codegen/test_reduce_sunmmio.py
  done

"""

import os
import re

import pytest
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from compile_pipeline import compile_test
from sunmmio_test_utils import save_final_ast, save_final_mlir, verify_final_mlir
from tilelang.utils.target import determine_target

import tvm

tilelang.env.disable_cache()


# (Shape, ReduceAxis, ExpectedInTileReduce, Clear)
REDUCE_TEST_CASES = [
    ((1024,), 0, True, True),
    ((32, 1024), 1, True, False),
    ((256, 128), 1, True, True),
    ((256, 128), 0, True, False),
    ((32, 256, 128), 2, True, True),
    ((32, 256, 128), 1, True, False),
    ((32, 256, 128), 0, False, True),
    ((32, 32, 256, 128), 3, True, False),
    ((32, 32, 256, 128), 1, False, True),
    ((32, 32, 32, 256, 128), 4, True, False),
    ((32, 32, 32, 256, 256), 0, False, True),
]


def parse_bool_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in ("1", "true", "t", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def clear_label(clear):
    return "clear_true" if clear else "clear_false"


def apply_reduce_op(reduce_op, buffer, out, reduce_axis, clear):
    if reduce_op == "sum":
        T.reduce_sum(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "abssum":
        T.reduce_abssum(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "max":
        T.reduce_max(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "absmax":
        T.reduce_absmax(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "min":
        T.reduce_min(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "bitand":
        T.reduce_bitand(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "bitor":
        T.reduce_bitor(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "bitxor":
        T.reduce_bitxor(buffer, out, dim=reduce_axis, clear=clear)
    else:
        raise ValueError(f"Unsupported TL_REDUCE_OP={reduce_op!r}")


def reduce_kernel_builder(shape, reduce_axis, dtype="float16", clear=True, reduce_op="sum"):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            apply_reduce_op(reduce_op, A_shared, Out_shared, reduce_axis, clear)
            T.copy(Out_shared, Out)

    return main


def reduce_tiled_test(
    B,
    M,
    N,
    block_B,
    block_M,
    block_N,
    tile_size,
    index_map,
    reduce_axis,
    dtype="float16",
    clear=False,
    reduce_op="abssum",
):
    out_shape_full = (B, M) if reduce_axis == 2 else (B, N) if reduce_axis == 1 else (M, N)
    out_shape_block = (block_B, block_M) if reduce_axis == 2 else (block_B, block_N) if reduce_axis == 1 else (block_M, block_N)
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols
    grid_b = T.ceildiv(B, block_B)
    grid_m = T.ceildiv(M, block_M)
    grid_n = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((B, M, N), dtype), Out: T.Tensor(out_shape_full, dtype)):
        with T.Kernel(ncores) as _cid:
            A_shared = T.alloc_shared((block_B, block_M, block_N), dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape_block, dtype, scope="shared.rsram")

            if reduce_axis == 2:
                for bz in T.serial(grid_b):
                    for by in T.serial(grid_m):
                        if (bz * grid_m + by) % ncores == _cid:
                            if clear:
                                T.fill(Out_shared, 0)
                            else:
                                T.copy(
                                    Out[
                                        bz * block_B : (bz + 1) * block_B,
                                        by * block_M : (by + 1) * block_M,
                                    ],
                                    Out_shared,
                                )
                            for bx in T.serial(grid_n):
                                T.copy(
                                    A[
                                        bz * block_B : (bz + 1) * block_B,
                                        by * block_M : (by + 1) * block_M,
                                        bx * block_N : (bx + 1) * block_N,
                                    ],
                                    A_shared,
                                )
                                apply_reduce_op(reduce_op, A_shared, Out_shared, reduce_axis, clear=False)
                            T.copy(
                                Out_shared,
                                Out[
                                    bz * block_B : (bz + 1) * block_B,
                                    by * block_M : (by + 1) * block_M,
                                ],
                            )
            elif reduce_axis == 1:
                for bz in T.serial(grid_b):
                    for bx in T.serial(grid_n):
                        if (bz * grid_n + bx) % ncores == _cid:
                            if clear:
                                T.fill(Out_shared, 0)
                            else:
                                T.copy(
                                    Out[
                                        bz * block_B : (bz + 1) * block_B,
                                        bx * block_N : (bx + 1) * block_N,
                                    ],
                                    Out_shared,
                                )
                            for by in T.serial(grid_m):
                                T.copy(
                                    A[
                                        bz * block_B : (bz + 1) * block_B,
                                        by * block_M : (by + 1) * block_M,
                                        bx * block_N : (bx + 1) * block_N,
                                    ],
                                    A_shared,
                                )
                                apply_reduce_op(reduce_op, A_shared, Out_shared, reduce_axis, clear=False)
                            T.copy(
                                Out_shared,
                                Out[
                                    bz * block_B : (bz + 1) * block_B,
                                    bx * block_N : (bx + 1) * block_N,
                                ],
                            )
            else:
                for by in T.serial(grid_m):
                    for bx in T.serial(grid_n):
                        if (by * grid_n + bx) % ncores == _cid:
                            if clear:
                                T.fill(Out_shared, 0)
                            else:
                                T.copy(
                                    Out[
                                        by * block_M : (by + 1) * block_M,
                                        bx * block_N : (bx + 1) * block_N,
                                    ],
                                    Out_shared,
                                )
                            for bz in T.serial(grid_b):
                                T.copy(
                                    A[
                                        bz * block_B : (bz + 1) * block_B,
                                        by * block_M : (by + 1) * block_M,
                                        bx * block_N : (bx + 1) * block_N,
                                    ],
                                    A_shared,
                                )
                                apply_reduce_op(reduce_op, A_shared, Out_shared, reduce_axis, clear=False)
                            T.copy(
                                Out_shared,
                                Out[
                                    by * block_M : (by + 1) * block_M,
                                    bx * block_N : (bx + 1) * block_N,
                                ],
                            )

    return main


def build_case():
    case = os.environ.get("TL_REDUCE_CASE", "tiled3d")
    dtype = os.environ.get("TL_REDUCE_DTYPE", "float16")
    reduce_op_env = os.environ.get("TL_REDUCE_OP")

    if case == "tiled3d":
        default_reduce_op = "abssum"
        reduce_op = reduce_op_env or default_reduce_op
        B, M, N = 64, 512, 1024
        block_B, block_M, block_N = 32, 256, 128
        tile_size = (4, 32)
        index_map = (-2, -1)
        reduce_axis = int(os.environ.get("TL_REDUCE_AXIS", "1"))
        clear = parse_bool_env("TL_REDUCE_CLEAR", False)
        func = reduce_tiled_test(
            B,
            M,
            N,
            block_B,
            block_M,
            block_N,
            tile_size,
            index_map,
            reduce_axis=reduce_axis,
            dtype=dtype,
            clear=clear,
            reduce_op=reduce_op,
        )
        op_suffix = f"_{reduce_op}" if reduce_op_env is not None else ""
        label = f"tiled3d{op_suffix}_axis{reduce_axis}_{clear_label(clear)}"
        expected_in_tile = reduce_axis in (1, 2)
        return label, func, expected_in_tile, clear

    default_reduce_op = "sum"
    reduce_op = reduce_op_env or default_reduce_op
    case_index = int(case)
    shape, reduce_axis, expected_in_tile, clear = REDUCE_TEST_CASES[case_index]
    clear = parse_bool_env("TL_REDUCE_CLEAR", clear)
    func = reduce_kernel_builder(shape, reduce_axis, dtype=dtype, clear=clear, reduce_op=reduce_op)
    op_suffix = f"_{reduce_op}" if reduce_op_env is not None else ""
    label = f"shape{case_index}{op_suffix}_axis{reduce_axis}_{'x'.join(map(str, shape))}_{clear_label(clear)}"
    return label, func, expected_in_tile, clear


PYTEST_TILED_REDUCE_CONFIG = dict(
    B=32,
    M=256,
    N=128,
    block_B=32,
    block_M=256,
    block_N=128,
    tile_size=(4, 32),
    index_map=(-2, -1),
    reduce_axis=1,
    clear=False,
)

PREVIOUS_OUTPUT_ABS_RE = re.compile(
    r"suvm\.tile\.load .*-> !suvm\.tile<1x32x[^>]+>\n"
    r"\s+%[0-9]+ = suvm\.tile\.abs %[0-9]+ : !suvm\.tile<1x32x[^>]+> -> !suvm\.tile<1x32x[^>]+>",
    re.MULTILINE,
)


def build_pytest_tiled_reduce_case(reduce_op, dtype):
    return reduce_tiled_test(
        reduce_op=reduce_op,
        dtype=dtype,
        **PYTEST_TILED_REDUCE_CONFIG,
    )


def compile_case(func, log_dir, run_codegen=True):
    log_dir = os.fspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    host_mod, device_mod = compile_test(
        func,
        out_idx=[1],
        target="Sunmmio",
        log_pass_output=True,
        log_dir=log_dir,
    )
    save_final_ast(log_dir, device_mod, echo=False)

    result = {
        "host_mod": host_mod,
        "device_mod": device_mod,
        "log_dir": log_dir,
    }

    if not run_codegen:
        return result

    target = determine_target("Sunmmio", return_object=True)
    mlir_mod = tvm.get_global_func("target.build.tilelang_sunmmio_without_compile")(device_mod, target, "suvm")
    mlir_source = mlir_mod.inspect_source()
    mlir_path = save_final_mlir(log_dir, mlir_source, echo=False)
    verify_final_mlir(log_dir, mlir_path)
    result["mlir_source"] = mlir_source
    result["mlir_path"] = mlir_path
    return result


@pytest.mark.parametrize(
    ("reduce_op", "dtype", "reduce_marker", "merge_marker"),
    [
        ("sum", "float16", "suvm.tile.reduce  sum", "suvm.tile.addf"),
        ("max", "float16", "suvm.tile.reduce  max", "suvm.tile.maxf"),
        ("min", "float16", "suvm.tile.reduce  min", "suvm.tile.minf"),
    ],
)
def test_tiled_clear_false_supported_reductions(tmp_path, reduce_op, dtype, reduce_marker, merge_marker):
    result = compile_case(
        build_pytest_tiled_reduce_case(reduce_op=reduce_op, dtype=dtype),
        tmp_path / f"{reduce_op}_{dtype}",
    )
    mlir_source = result["mlir_source"]
    assert reduce_marker in mlir_source
    assert merge_marker in mlir_source
    assert PREVIOUS_OUTPUT_ABS_RE.search(mlir_source) is None


@pytest.mark.parametrize(
    ("reduce_op", "dtype"),
    [
        ("abssum", "float16"),
        ("absmax", "float16"),
    ],
)
def test_tiled_clear_false_absolute_reductions_should_not_abs_previous_output(tmp_path, reduce_op, dtype):
    result = compile_case(
        build_pytest_tiled_reduce_case(reduce_op=reduce_op, dtype=dtype),
        tmp_path / f"{reduce_op}_{dtype}",
    )
    assert PREVIOUS_OUTPUT_ABS_RE.search(result["mlir_source"]) is None


@pytest.mark.parametrize("reduce_op", ["bitand", "bitor", "bitxor"])
def test_tiled_clear_false_bitwise_reductions_should_codegen_for_int32(tmp_path, reduce_op):
    result = compile_case(
        build_pytest_tiled_reduce_case(reduce_op=reduce_op, dtype="int32"),
        tmp_path / f"{reduce_op}_int32",
    )
    assert "suvm.tile.reduce" in result["mlir_source"]


def main():
    label, func, expected_in_tile, clear = build_case()
    log_root = os.environ.get(
        "TL_REDUCE_LOG_ROOT",
        os.path.join(os.path.dirname(__file__), "logs_reduce_sunmmio"),
    )
    log_dir = os.path.join(log_root, label)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n--- Testing Reduce Case: {label} ---")
    print(f"Expected vector_core_in_tile_reduce: {expected_in_tile}")
    print(f"Reduce clear: {clear}")
    func.show()

    host_mod, device_mod = compile_test(
        func,
        out_idx=[1],
        target="Sunmmio",
        log_pass_output=True,
        log_dir=log_dir,
    )
    device_mod.show()
    save_final_ast(log_dir, device_mod, echo=True)

    if os.environ.get("TL_REDUCE_SKIP_CODEGEN", "0") == "1":
        return

    print("--- Running CodeGenTileLangSunMMIO ---")
    target = determine_target("Sunmmio", return_object=True)
    mlir_mod = tvm.get_global_func("target.build.tilelang_sunmmio_without_compile")(device_mod, target, "suvm")
    mlir_source = mlir_mod.inspect_source()
    mlir_path = save_final_mlir(log_dir, mlir_source, echo=True)
    verify_final_mlir(log_dir, mlir_path)


if __name__ == "__main__":
    main()
