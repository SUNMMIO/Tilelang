import argparse
from typing import Callable

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout


def ref_program(x):
    import numpy as np

    x = x.astype("float32")
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_kernel(M, N, block_M, block_N, dtype: T.dtype = T.bfloat16) -> "Callable":
    zz_layout = make_zz_layout((M, N))
    placement = T.MeshShardingPolicy(y=0, x=1)

    accum_dtype = T.float32
    scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(X: T.MeshTensor((M, N), placement, dtype, layout=zz_layout), Y: T.MeshTensor((M, N), placement, dtype, layout=zz_layout)):
        with T.Kernel() as (_cid):
            sharded_M, sharded_N = X.local_shape

            X_shared = T.alloc_shared((block_M, block_N), dtype)
            Y_shared = T.alloc_shared((block_M, block_N), dtype)
            lse = T.alloc_shared((block_M), accum_dtype)
            max_x = T.alloc_shared((block_M,), dtype)
            exp_x = T.alloc_shared([block_M, block_N], accum_dtype)
            sum_exp_x = T.alloc_shared((block_M), accum_dtype)

            lse_dist = T.alloc_shared((T.mesh_ncols(), block_M), accum_dtype)
            lse_max = T.alloc_shared((block_M,), accum_dtype)
            lse_global = T.alloc_shared((block_M,), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                T.fill(lse, -T.infinity(accum_dtype))

                # Could be pipelined
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(X[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], X_shared)
                    T.reduce_max(X_shared, max_x, dim=-1, clear=True)
                    for i, j in T.Tiles([block_M, block_N]):
                        exp_x[i, j] = T.exp2(X_shared[i, j] * scale - max_x[i] * scale)
                    T.reduce_sum(exp_x, sum_exp_x, dim=-1, clear=True)

                    for i in T.Tiles([block_M]):
                        lse[i] = max_x[i] * scale + T.log2(T.exp2(lse[i] - max_x[i] * scale) + sum_exp_x[i])

                # Aggregate
                T.comm.all_gather(lse, lse_dist, direction="h")
                T.reduce_max(lse_dist, lse_max, dim=0, clear=True)

                # Get global lse
                for i, j in T.Tiles([T.mesh_ncols(), block_M]):
                    lse_dist[i, j] = T.exp2(lse_dist[i, j] - lse_max[j])
                T.reduce_sum(lse_dist, lse_global, dim=0, clear=True)
                for i in T.Tiles([block_M]):
                    lse_global[i] = lse_max[i] + T.log2(lse_global[i])

                # Get results
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(X[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], X_shared)
                    for i, j in T.Tiles([block_M, block_N]):
                        Y_shared[i, j] = T.exp2(X_shared[i, j] * scale - lse_global[i])
                    T.copy(Y_shared, Y[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


@tilelang.jit(target="sunmmio", execution_backend="sunmmio_sunsim")
def online_softmax(M, N, block_M, block_N, dtype):
    return softmax_kernel(M, N, block_M, block_N, dtype)


def main(M, N) -> None:
    import ml_dtypes
    import numpy as np
    import sunsim

    rng = np.random.default_rng(0)
    x = rng.uniform(-4.0, 4.0, size=(M, N)).astype(np.float32).astype(ml_dtypes.bfloat16)

    kernel = online_softmax(
        M,
        N,
        block_M=256,
        block_N=256,
        dtype=T.bfloat16,
    )

    placement = [sunsim.S(0), sunsim.S(1)]
    layout = sunsim.Layout.zz(block_dims=(0, 1))
    y = sunsim.Output((M, N), ml_dtypes.bfloat16, placement=placement, layout=layout)
    result = kernel(
        sunsim.Input(x, placement=placement, layout=layout),
        y,
        mesh=driver.get_sunmmio_device_mesh_config(),
        timeout=1200.0,
    )

    np.testing.assert_allclose(y.data.astype(np.float32), ref_program(x), rtol=5e-2, atol=5e-3)
    print(f"online_softmax PASS: shape=({M}, {N}) matched reference within tolerance")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    args, _ = parser.parse_known_args()
    main(args.m, args.n)
