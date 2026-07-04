import argparse

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_target
from tilelang.layout import make_zz_layout, make_row_major


def reduction(M, K, N, block_K, block_N, in_dtype, out_dtype):
    zz_layout = make_zz_layout((M, K, N))
    placement = T.MeshShardingPolicy(y=0, x=1)
    rm_layout = make_row_major((M, K))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K, N), placement, in_dtype, layout=zz_layout),
        B: T.MeshTensor((M, K), placement, out_dtype, layout=rm_layout),
    ):
        with T.Kernel() as _cid:
            sharded_M, sharded_K, sharded_N = A.local_shape
            print(sharded_M, sharded_K, sharded_N)

            A_shared = T.alloc_shared((block_K, block_N), in_dtype)
            Acc_shared = T.alloc_shared((block_K, block_N), out_dtype)
            Acc_dist_shared = T.alloc_shared((block_K, T.mesh_ncols() * block_N), out_dtype)
            B_shared = T.alloc_shared((block_K,), out_dtype)

            for bx in T.serial(sharded_M):
                for by in T.serial(T.ceildiv(sharded_K, block_K)):
                    T.clear(Acc_shared)
                    for bz in T.serial(T.ceildiv(sharded_N, block_N)):
                        T.copy(A[bx, by * block_K, bz * block_N], A_shared)
                        for i, j in T.Tiles([block_K, block_N]):
                            Acc_shared[i, j] = A_shared[i, j] + Acc_shared[i, j]
                    T.comm.all_gather(Acc_shared, Acc_dist_shared, "h", axis=-1)
                    T.reduce_sum(Acc_dist_shared, B_shared, dim=-1)
                    T.copy(B_shared, B[bx, by * block_K])

    return main


def main(M, K, N) -> None:
    target = determine_target("Sunmmio", return_object=True)

    pass_configs = {tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True}
    with tvm.target.Target(target), tvm.transform.PassContext(config=pass_configs):
        kernel = reduction(M, K, N, block_K=128, block_N=128, in_dtype=T.bfloat16, out_dtype=T.float32)
        mod = LowerAndLegalize(tvm.IRModule({"main": kernel}), target)
        print(mod)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    args, _ = parser.parse_known_args()
    main(args.m, args.k, args.n)
