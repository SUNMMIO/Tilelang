"""Mixed BF16/FP32 T.Tiles example with exact sunsim verification.

The execution tile is inferred from the layouts of all three shared buffers.
The two inputs and multiplication use BF16; the output is stored as FP32.
"""

import argparse
from pathlib import Path

import tilelang
import tilelang.language as T


@tilelang.jit(target="sunmmio", execution_backend="sunmmio_sunsim")
def mixed_precision_mul(M=256, N=512, block_M=64, block_N=128):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def mixed_mul(
        A: T.MeshTensor((M, N), placement, T.bfloat16),
        B: T.MeshTensor((M, N), placement, T.bfloat16),
        C: T.MeshTensor((M, N), placement, T.float32),
    ):
        with T.Kernel():
            local_m, local_n = A.local_shape
            A_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
            B_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
            C_shared = T.alloc_shared((block_M, block_N), T.float32)
            for bm in T.serial(T.ceildiv(local_m, block_M)):
                for bn in T.serial(T.ceildiv(local_n, block_N)):
                    T.copy(A[bm * block_M, bn * block_N], A_shared)
                    T.copy(B[bm * block_M, bn * block_N], B_shared)
                    for i, j in T.Tiles(A_shared):
                        C_shared[i, j] = A_shared[i, j] * B_shared[i, j]
                    T.copy(C_shared, C[bm * block_M, bn * block_N])

    return mixed_mul


def main(M=256, N=512, block_M=64, block_N=128, working_dir=None):
    import ml_dtypes
    import numpy as np
    import sunsim

    rng = np.random.default_rng(20260922)
    a = rng.standard_normal((M, N)).astype(ml_dtypes.bfloat16)
    b = rng.standard_normal((M, N)).astype(ml_dtypes.bfloat16)
    placement = [sunsim.S(0), sunsim.S(1)]
    layout = sunsim.Layout.zz(block_dims=(0, 1))
    c = sunsim.Output((M, N), np.float32, placement=placement, layout=layout)
    kernel = mixed_precision_mul(M, N, block_M, block_N)
    result = kernel(
        sunsim.Input(a, placement=placement, layout=layout),
        sunsim.Input(b, placement=placement, layout=layout),
        c,
        working_dir=working_dir,
        timeout=240.0,
    )
    # TIR evaluates BF16 * BF16 in BF16 before widening the result for storage.
    np.testing.assert_array_equal(c.data, (a * b).astype(np.float32))
    assert result.exit_code == 0
    print(f"PASS: BF16 multiply -> FP32 storage, shape=({M}, {N})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--working-dir", type=Path)
    args = parser.parse_args()
    main(args.m, args.n, args.block_m, args.block_n, args.working_dir)
