import os
import re

import tilelang
import tilelang.language as T
import tilelang.testing

from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt
from testing.python.sunmmio.common.compile_pipeline import target


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")

LOOSE_OPT_ARGS = ("--verify-each",)


@target("Sunmmio")
def partitioned_view_region_kernel(
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
        with T.Kernel():
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, 1, block_N), accum_dtype)

            T.copy(A[128, 192], A_shared)
            T.copy(B[224, 256], B_shared)
            T.copy(C[32, 0, 64], C_shared[:block_M, 0, :block_N])

    return main


def _mlir_int_values(mlir_source: str) -> dict[str, int]:
    constants = {
        f"%{name}": int(value)
        for name, value in re.findall(
            r"%([A-Za-z0-9_]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*(?:i\d+|index)",
            mlir_source,
        )
    }
    aliases = {
        f"%{dst}": f"%{src}"
        for dst, src in re.findall(
            r"%([A-Za-z0-9_]+)\s*=\s*arith\.index_cast\s+%([A-Za-z0-9_]+)\s*:\s*i32\s+to\s+index",
            mlir_source,
        )
    }

    values = dict(constants)
    changed = True
    while changed:
        changed = False
        for dst, src in aliases.items():
            if dst not in values and src in values:
                values[dst] = values[src]
                changed = True
    return values


def _partitioned_view_indices(mlir_source: str) -> list[list[int]]:
    values = _mlir_int_values(mlir_source)
    indices = []

    for line in mlir_source.splitlines():
        if "suvm.get_partitioned_tile_view" not in line:
            continue
        match = re.search(r"indices\s*=\s*\[([^\]]*)\]", line)
        if not match:
            continue

        line_indices = []
        for token in (item.strip() for item in match.group(1).split(",")):
            if not token:
                continue
            if token in values:
                line_indices.append(values[token])
            elif re.fullmatch(r"-?\d+", token):
                line_indices.append(int(token))
            else:
                raise AssertionError(f"Could not resolve partitioned tile view index {token!r} in line: {line}")
        indices.append(line_indices)

    return indices


def test_region_call_partitioned_view_indices_are_tile_coordinates(tmp_path):
    src = validate_sunmmio_codegen_with_npuir_opt(
        partitioned_view_region_kernel(),
        tmp_path,
        mlir_filename="region_call_partitioned_view.mlir",
        expected_tokens=("suvm.copy_async", "suvm.get_partitioned_tile_view"),
        opt_args=LOOSE_OPT_ARGS,
    )

    # Nonzero element offsets should become tile coordinates:
    # A[128, 192] -> [4, 6], B[224, 256] -> [7, 8],
    # C[32, 0, 64] -> [1, 0, 2].
    indices = _partitioned_view_indices(src)
    assert [4, 6] in indices
    assert [7, 8] in indices
    assert [1, 0, 2] in indices
    assert [128, 192] not in indices
    assert [224, 256] not in indices
    assert [32, 0, 64] not in indices
    assert "arith.divsi" not in src


if __name__ == "__main__":
    tilelang.testing.main()
