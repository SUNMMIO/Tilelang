import os
import re
import tilelang
import tilelang.language as T
import tilelang.testing

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"

LOOSE_OPT_ARGS = ("--verify-each",)


@target("Sunmmio")
def region_test_kernel(
    M=1024,
    N=1024,
    K=1024,
    block_M=32,
    block_N=32,
    block_K=32,
    dtype="float16",
    accum_dtype="float32",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N, K), accum_dtype),
    ):
        with T.Kernel(1):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, 1, block_N), accum_dtype)

            T.copy(A[128, 192], A_shared)
            T.copy(B[224, 256], B_shared)
            T.copy(C[32, 0, 64], C_shared[:block_M, 0, :block_N])

    return main


def extract_value_mapping(mlir_source: str) -> dict:
    """
    从 MLIR 源代码中提取所有值的映射关系。
    处理两种映射：
    1. arith.constant: %c7_i32 = arith.constant 7 : i32 -> {"%c7_i32": 7}
    2. arith.index_cast: %13 = arith.index_cast %c7_i32 : i32 to index -> {"%13": "%c7_i32"}

    最终返回从最终变量名到整数值的映射。
    """
    # 第一步：提取所有常量值
    constant_map = {}
    pattern_const = r"%([a-zA-Z0-9_]+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*i32"
    for name, value in re.findall(pattern_const, mlir_source):
        constant_map[f"%{name}"] = int(value)

    # 第二步：提取所有 index_cast 映射
    # 格式: %13 = arith.index_cast %c7_i32 : i32 to index
    pattern_cast = r"%([a-zA-Z0-9_]+)\s*=\s*arith\.index_cast\s+%([a-zA-Z0-9_]+)\s*:\s*i32\s+to\s+index"

    # 第三步：解析映射链，从 %13 -> %c7_i32 -> 7
    # 先建立 cast 映射
    cast_map = {}  # "%13" -> "%c7_i32"
    for dst, src in re.findall(pattern_cast, mlir_source):
        cast_map[f"%{dst}"] = f"%{src}"

    # 第四步：解析最终的映射（从 cast 目标变量到整数值）
    result_map = {}
    for dst, src in cast_map.items():
        if src in constant_map:
            result_map[dst] = constant_map[src]

    # 也把常量本身加入结果
    result_map.update(constant_map)

    return result_map


def extract_partitioned_view_indices(mlir_source: str) -> list:
    """
    从 MLIR 源代码中提取所有 suvm.get_partitioned_tile_view 的 indices 值。
    返回一个列表，每个元素是 indices 的整数值列表。
    """
    indices_list = []

    # 先提取所有常量值
    value_map = extract_value_mapping(mlir_source)

    patterns = [
        r"suvm\.get_partitioned_tile_view\s+[^,]+,\s*indices\s*=\s*\[([^\]]+)\]",
        r"suvm\.get_partitioned_tile_view\s+[^,[]+\s*indices\s*=\s*\[([^\]]+)\]",
        r"get_partitioned_tile_view[^\]]*indices\s*=\s*\[([^\]]+)\]",
    ]

    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, mlir_source)
        all_matches.extend(matches)

    # 去重
    seen = set()
    unique_matches = []
    for match in all_matches:
        if match not in seen:
            seen.add(match)
            unique_matches.append(match)

    for match in unique_matches:
        # match 是类似 "%13, %14" 或 "%29, %30, %31" 的字符串
        # 提取所有 %name
        var_names = re.findall(r"%([a-zA-Z0-9_]+)", match)

        indices = []
        for var_name in var_names:
            key = f"%{var_name}"
            if key in value_map:
                indices.append(value_map[key])
            else:
                # 如果找不到，尝试直接解析数字
                try:
                    indices.append(int(var_name))
                except ValueError:
                    print(f"Warning: Cannot resolve value for {key}")
                    indices.append(-1)

        # 如果 var_names 为空，尝试直接解析数字
        if not var_names:
            numbers = re.findall(r"\b(\d+)\b", match)
            indices = [int(n) for n in numbers]

        if indices:
            indices_list.append(indices)

    return indices_list


def validate_partitioned_view_indices(
    mlir_source: str,
    expected_indices_list: list,
) -> bool:
    """
    校验 get_partitioned_tile_view 的 indices 值。

    Args:
        mlir_source: MLIR 源代码
        expected_indices_list: 期望的 indices 列表

    Returns:
        bool: 所有校验通过返回 True
    """
    all_passed = True

    # 提取实际的 indices
    actual_indices_list = extract_partitioned_view_indices(mlir_source)
    # 舍弃 suvm.transform_layout_async 中的最后两个
    actual_indices_list = actual_indices_list[:-2]

    # 校验每个 indices
    for i, (actual, expected) in enumerate(zip(actual_indices_list, expected_indices_list)):
        print(f"Call {i}: actual = {actual}, expected = {expected}")

        if actual != expected:
            print(f"  ❌ MISMATCH: {actual} != {expected}")
            all_passed = False
        else:
            print("  ✅ MATCH")

    return all_passed


def test_simple_global_copy_gemm_codegen_validates_with_npuir_opt(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        region_test_kernel(),
        tmp_path,
        mlir_filename="region_test_kernel.mlir",
        expected_tokens=("suvm.copy_async",),
        opt_args=LOOSE_OPT_ARGS,
    )

    print(src)

    src_lines = src.split("\n")
    src_lines = [it.strip() for it in src_lines]

    # 提取所有常量值，用于调试
    print("\n" + "=" * 60)
    print("Extracted Constants:")
    print("=" * 60)
    constants = extract_value_mapping(src)
    for name, value in constants.items():
        print(f"  {name} = {value}")

    # 校验 get_partitioned_tile_view 的 indices
    # A[128, 192] -> block_size=32 -> [128/32=4, 192/32=6]
    # B[224, 256] -> block_size=32 -> [224/32=7, 256/32=8]
    # C[32, 0, 64] -> block_size=32 -> [32/32=1, 0/32=0, 64/32=2]
    expected_indices = [
        [4, 6],  # A[128, 192]
        [0, 0],  # A_shared
        [7, 8],  # B[224, 256]
        [0, 0],  # B_shared
        [1, 0, 2],  # C[32, 0, 64]
        [0, 0, 0],  # C_shared[:block_M, 0, :block_N])
    ]

    print("\n" + "=" * 60)
    print("Validating get_partitioned_tile_view indices:")
    print("=" * 60)
    result = validate_partitioned_view_indices(src, expected_indices)

    if result:
        print("\n✅ All get_partitioned_view indices validated successfully!")
    else:
        print("\n❌ Some get_partitioned_view indices validation failed!")

    assert result, "get_partitioned_view indices validation failed"

    # 打印所有包含 get_partitioned_tile_view 的行
    print("\n" + "=" * 60)
    print("All lines containing 'get_partitioned_tile_view':")
    print("=" * 60)
    for line in src_lines:
        if "get_partitioned_tile_view" in line:
            print(line)


if __name__ == "__main__":
    tilelang.testing.main()
