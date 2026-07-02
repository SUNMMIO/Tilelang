import argparse
from typing import Callable

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.carver.arch import driver
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_target
from tilelang.layout import make_zz_layout


def softmax_kernel(M, N, block_M, block_N, dtype: T.dtype = T.bfloat16) -> "Callable":
    mesh = driver.get_sunmmio_device_mesh_config()
    _, ncols = mesh

    zz_layout = make_zz_layout((M, N))
    placement = T.MeshShardingPolicy(y=0, x=1)

    accum_dtype = T.float32
    scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(X: T.MeshTensor((M, N), placement, mesh, dtype, zz_layout), Y: T.MeshTensor((M, N), placement, mesh, dtype, zz_layout)):
        with T.Kernel() as (_cid):
            sharded_M, sharded_N = X.local_shape

            X_shared = T.alloc_shared((block_M, block_N), dtype)
            Y_shared = T.alloc_shared((block_M, block_N), dtype)
            lse = T.alloc_shared((block_M), accum_dtype)
            max_x = T.alloc_shared((block_M,), dtype)
            exp_x = T.alloc_shared([block_M, block_N], accum_dtype)
            sum_exp_x = T.alloc_shared((block_M), accum_dtype)

            lse_dist = T.alloc_shared((ncols, block_M), accum_dtype)
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
                for i, j in T.Tiles([ncols, block_M]):
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


def main(M, N) -> None:
    target = determine_target("Sunmmio", return_object=True)

    pass_configs = {tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True}
    with tvm.target.Target(target), tvm.transform.PassContext(config=pass_configs):
        kernel = softmax_kernel(M, N, block_M=128, block_N=128, dtype=T.bfloat16)
        mod = LowerAndLegalize(tvm.IRModule({"main": kernel}), target)
        print(mod)

    # ELF

    # Prepare data (Torch CPU)

    # To torch sunmmio Sunmmio A4E zpu (Torch Sunmmio)
    # Launch & Get Result (Compile & Torch Sunmmio Execution)
    # SuDeck

    # reference program (Torch CPU)

    # Torch Sunmmio Tensor -> Torch CPU Tensor
    # Compare (Torch CPU)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    args, _ = parser.parse_known_args()
    main(args.m, args.n)
