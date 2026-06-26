import argparse
from typing import Callable

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.carver.arch import driver
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_target
from tilelang.layout import make_zz_layout


def rmsnorm_kernel(M, N, block_M, block_N, dtype: T.dtype = T.bfloat16, eps: float = 1e-12) -> "Callable":
    # Device configuration: a row x col mesh of cores.
    mesh = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = mesh
    ncores = nrows * ncols

    zz_layout = make_zz_layout((M, N))
    # Shard rows (dim 0) across the mesh rows and the reduced dim N (dim 1)
    # across the mesh columns, mirroring the softmax example.  The RMSNorm
    # reduction is over N, which lives on the column axis, so each core holds a
    # partial sum that is combined across the row with an all_gather.
    placement = T.MeshShardingPolicy(y=0, x=1)

    accum_dtype = T.float32

    @T.prim_func
    def main(
        X: T.MeshTensor((M, N), placement, mesh, dtype, zz_layout),
        Y: T.MeshTensor((M, N), placement, mesh, dtype, zz_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_N = X.shape

            X_shared = T.alloc_shared((block_M, block_N), dtype)
            Y_shared = T.alloc_shared((block_M, block_N), dtype)
            x_sq = T.alloc_shared((block_M, block_N), accum_dtype)
            tile_sumsq = T.alloc_shared((block_M,), accum_dtype)
            local_sumsq = T.alloc_shared((block_M,), accum_dtype)

            sumsq_dist = T.alloc_shared((ncols, block_M), accum_dtype)
            total_sumsq = T.alloc_shared((block_M,), accum_dtype)
            inv_rms = T.alloc_shared((block_M,), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                T.fill(local_sumsq, 0)

                # Local partial sum of squares over this core's N shard.
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(X[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], X_shared)
                    for i, j in T.Tiles([block_M, block_N]):
                        x_sq[i, j] = X_shared[i, j].astype(accum_dtype) * X_shared[i, j].astype(accum_dtype)
                    T.reduce_sum(x_sq, tile_sumsq, dim=-1, clear=True)
                    for i in T.Tiles([block_M]):
                        local_sumsq[i] = local_sumsq[i] + tile_sumsq[i]

                # Combine the partial sums across the row (the N-sharding axis).
                T.comm.all_gather(local_sumsq, sumsq_dist, direction="h")
                T.reduce_sum(sumsq_dist, total_sumsq, dim=0, clear=True)

                # inv_rms = rsqrt(mean(x^2) + eps), with the mean over the full N.
                for i in T.Tiles([block_M]):
                    inv_rms[i] = T.rsqrt(total_sumsq[i] / N + eps)

                # Scale and write back.
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(X[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], X_shared)
                    for i, j in T.Tiles([block_M, block_N]):
                        Y_shared[i, j] = X_shared[i, j] * inv_rms[i]
                    T.copy(Y_shared, Y[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


def main(M, N) -> None:
    target = determine_target("Sunmmio", return_object=True)

    pass_configs = {tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True}
    with tvm.target.Target(target), tvm.transform.PassContext(config=pass_configs):
        kernel = rmsnorm_kernel(M, N, block_M=128, block_N=128, dtype=T.bfloat16)
        mod = LowerAndLegalize(tvm.IRModule({"main": kernel}), target)
        print(mod)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    args, _ = parser.parse_known_args()
    main(args.m, args.n)
