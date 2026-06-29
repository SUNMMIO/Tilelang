import argparse

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.carver.arch import driver
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_target
from tilelang.layout import make_row_major, make_aligned_row_major


def reduction(M, block_M, in_dtype, out_dtype):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    A_layout = make_row_major((M,))
    B_layout = make_aligned_row_major((1,), align_bytes=1024, dtype=out_dtype)
    A_placement = T.MeshShardingPolicy(y=0, replicate=T.MeshReplicationType.ROW)
    B_placement = T.MeshShardingPolicy(replicate=T.MeshReplicationType.ALL)

    @T.prim_func
    def main(
        A: T.MeshTensor((M), A_placement, device_mesh_config, in_dtype, layout=A_layout),
        B: T.MeshTensor((1), B_placement, device_mesh_config, out_dtype, layout=B_layout),
    ):
        with T.Kernel(ncores) as _cid:
            sharded_M = A.shape[0]

            A_shared = T.alloc_shared((block_M), in_dtype)
            Acc_shared = T.alloc_shared((block_M), out_dtype)
            Acc_dist_shared = T.alloc_shared((ncols * block_M), out_dtype)
            B_shared = T.alloc_shared((1,), out_dtype)
            T.annotate_layout({B_shared: B_layout})

            T.clear(Acc_shared)
            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                T.copy(A[bx * block_M : (bx + 1) * block_M], A_shared)
                for i in T.Tiles([block_M]):
                    Acc_shared[i] = A_shared[i] + Acc_shared[i]
            T.comm.all_gather(Acc_shared, Acc_dist_shared, "h", axis=-1)
            T.reduce_sum(Acc_dist_shared, B_shared, dim=-1)
            T.copy(B_shared, B)

    return main


def main(M) -> None:
    target = determine_target("Sunmmio", return_object=True)

    pass_configs = {tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True}
    with tvm.target.Target(target), tvm.transform.PassContext(config=pass_configs):
        kernel = reduction(M, block_M=128, in_dtype=T.bfloat16, out_dtype=T.bfloat16)
        mod = LowerAndLegalize(tvm.IRModule({"main": kernel}), target)
        print(mod)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    args, _ = parser.parse_known_args()
    main(args.m)
