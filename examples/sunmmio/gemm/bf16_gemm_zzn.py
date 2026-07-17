import argparse

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import make_nzz_layout, make_zz_layout


def ref_program(a, b):
    return a.astype("float32") @ b.astype("float32")


def _validate_config(M, N, K, block_M, block_N, block_K):
    mesh_rows, mesh_cols = driver.get_sunmmio_device_mesh_config()
    asram_bytes = block_M * (mesh_cols * block_K) * 2
    wsram_bytes = (mesh_rows * block_K) * block_N * 2

    if min(M, N, K, block_M, block_N, block_K) <= 0:
        raise ValueError("matrix and block dimensions must be positive")
    if block_M % 32 != 0 or block_N % 32 != 0 or block_K % 32 != 0:
        raise ValueError("block_M, block_N, and block_K must be multiples of 32")
    if asram_bytes > 128 * 1024:
        raise ValueError(f"A tile requires {asram_bytes} ASRAM bytes, exceeding the 128 KiB limit")
    if wsram_bytes > 1024 * 1024:
        raise ValueError(f"B tile requires {wsram_bytes} WSRAM bytes, exceeding the 1 MiB limit")
    sharded_M = ((M + mesh_rows * 32 - 1) // (mesh_rows * 32)) * 32
    if block_M > sharded_M:
        raise ValueError(f"block_M must not exceed the ZZ-padded per-core M extent ({sharded_M})")
    if K % (mesh_rows * block_K) != 0:
        raise ValueError(f"K must be a multiple of mesh_rows * block_K ({mesh_rows * block_K})")
    if N % (mesh_cols * block_N) != 0:
        raise ValueError(f"N must be a multiple of mesh_cols * block_N ({mesh_cols * block_N})")


def matmul_persistent_zzn(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
    _validate_config(M, N, K, block_M, block_N, block_K)

    A_layout = make_zz_layout((M, K))
    B_layout = make_nzz_layout(
        (K, N),
        axes=(0, 1),
        block_shape=(32, 32),
        cluster_shape=(block_K // 32, block_N // 32),
    )
    C_layout = make_zz_layout((M, N))
    placement = T.MeshShardingPolicy(y=0, x=1)

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), placement, dtype, layout=A_layout),
        B: T.MeshTensor((K, N), placement, dtype, layout=B_layout),
        C: T.MeshTensor((M, N), placement, accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as _cid:
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape

            A_shared_dist = T.alloc_shared((block_M, block_K * T.mesh_ncols()), dtype)
            B_shared_dist = T.alloc_shared((block_K * T.mesh_nrows(), block_N), dtype)
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
def bf16_gemm_zzn(M, N, K, block_M, block_N, block_K):
    return matmul_persistent_zzn(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        dtype=T.bfloat16,
        accum_dtype=T.float32,
    )


def main(M=256, N=256, K=256, block_M=32, block_N=64, block_K=64, timeout=720.0):
    import ml_dtypes
    import numpy as np
    import sunsim

    _validate_config(M, N, K, block_M, block_N, block_K)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float32).astype(ml_dtypes.bfloat16)
    b = rng.standard_normal((K, N)).astype(np.float32).astype(ml_dtypes.bfloat16)

    kernel = bf16_gemm_zzn(
        M,
        N,
        K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )

    placement = [sunsim.S(0), sunsim.S(1)]
    zz_layout = sunsim.Layout.zz(block_dims=(0, 1))
    zzn_layout = sunsim.Layout.zzn(block_dims=(0, 1), super_size=(block_K, block_N))
    c = sunsim.Output((M, N), np.float32, placement=placement, layout=zz_layout)
    result = kernel(
        sunsim.Input(a, placement=placement, layout=zz_layout),
        sunsim.Input(b, placement=placement, layout=zzn_layout),
        c,
        mesh=driver.get_sunmmio_device_mesh_config(),
        timeout=timeout,
    )

    np.testing.assert_allclose(c.data, ref_program(a, b), rtol=1e-2, atol=1e-2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=720.0)
    args, _ = parser.parse_known_args()
    main(args.m, args.n, args.k, args.block_m, args.block_n, args.block_k, args.timeout)
