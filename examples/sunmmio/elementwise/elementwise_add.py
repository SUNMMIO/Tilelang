import argparse

import tilelang
from tilelang import tvm as tvm
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_target


def elementwise_add(M, N, block_M, block_N, in_dtype, out_dtype):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    zz_layout = make_zz_layout((M, N))
    placement = T.MeshShardingPolicy(y=0, x=1)

    nrows = T.symbolic("nrows")
    ncols = T.symbolic("ncols")
    ncores = nrows * ncols

    @T.prim_func
    def elem_add(
        A: T.MeshTensor((M, N), placement, device_mesh_config, in_dtype, layout=zz_layout),
        B: T.MeshTensor((M, N), placement, device_mesh_config, in_dtype, layout=zz_layout),
        C: T.MeshTensor((M, N), placement, device_mesh_config, out_dtype, layout=zz_layout),
    ):
        with T.Kernel(ncores) as _cid:
            sharded_M, sharded_N = A.shape

            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(A[bx * block_M, by * block_N], A_shared)
                    T.copy(B[bx * block_M, by * block_N], B_shared)
                    for i, j in T.Tiles([block_M, block_N]):
                        C_shared[i, j] = A_shared[i, j] + B_shared[i, j]
                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return elem_add


def main(M, N) -> None:
    target = determine_target("Sunmmio", return_object=True)

    # Enable layout visualization. With this pass config set, LowerAndLegalize
    # prints each buffer's inferred CuteLayout (the on-chip layout_map and the
    # DRAM global_layout_map) right after Sunmmio layout inference, via
    # tilelang.analysis.SunmmioLayoutVisual. CuteLayouts are textual only, so the
    # TL_LAYOUT_VISUALIZATION_FORMATS option (for CUDA fragment diagrams) is N/A here.
    pass_configs = {tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True}
    with tvm.target.Target(target), tvm.transform.PassContext(config=pass_configs):
        kernel = elementwise_add(M, N, block_M=32, block_N=32, in_dtype=T.bfloat16, out_dtype=T.float32)
        # Buffer layouts are printed during this call; the lowered IR follows.
        mod = LowerAndLegalize(tvm.IRModule({"elem_add": kernel}), target)
        print(mod)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    args, _ = parser.parse_known_args()
    main(args.m, args.n)
