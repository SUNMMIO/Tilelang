import argparse

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout


def ref_program(a, b):
    return a.astype("float32") @ b.astype("float32")


def matmul_persistent(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
    A_layout = make_zz_layout((M, K))
    B_layout = make_zz_layout((K, N))
    C_layout = make_zz_layout((M, N))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.placement.full_shard(0, 1), dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.placement.full_shard(0, 1), accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape

            A_shared_dist = T.alloc_shared((block_M, block_K * T.ncols()), dtype)
            B_shared_dist = T.alloc_shared((block_K * T.nrows(), block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_shared)
                    for k in T.serial(T.ceildiv(sharded_K, block_K)):
                        T.comm.all_gather(
                            A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                            A_shared_dist,
                            direction="horizontal",
                            axis=-1,
                        )
                        T.comm.all_gather(
                            B[k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N],
                            B_shared_dist,
                            direction="vertical",
                            axis=0,
                        )
                        T.gemm(A_shared_dist, B_shared_dist, C_shared)
                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return main


@tilelang.jit(target="sunmmio", execution_backend="sunmmio_sunsim")
def bf16_gemm(M, N, K, block_M, block_N, block_K):
    return matmul_persistent(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        dtype=T.bfloat16,
        accum_dtype=T.float32,
    )


def main(M=128, N=128, K=128, block_M=32, block_N=32, block_K=32, timeout=240.0):
    import ml_dtypes
    import numpy as np
    import sunsim

    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float32).astype(ml_dtypes.bfloat16)
    b = rng.standard_normal((K, N)).astype(np.float32).astype(ml_dtypes.bfloat16)

    kernel = bf16_gemm(
        M,
        N,
        K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )

    placement = [sunsim.S(0), sunsim.S(1)]
    layout = sunsim.Layout.zz(block_dims=(0, 1))
    c = sunsim.Output((M, N), np.float32, placement=placement, layout=layout)
    result = kernel(
        sunsim.Input(a, placement=placement, layout=layout),
        sunsim.Input(b, placement=placement, layout=layout),
        c,
        mesh=driver.get_sunmmio_device_mesh_config(),
        timeout=timeout,
    )

    np.testing.assert_allclose(c.data, ref_program(a, b), rtol=1e-2, atol=1e-2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=240.0)
    args, _ = parser.parse_known_args()
    main(args.m, args.n, args.k, args.block_m, args.block_n, args.block_k, args.timeout)
