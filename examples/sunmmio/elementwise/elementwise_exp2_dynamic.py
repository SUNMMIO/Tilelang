import argparse

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout


def ref_program(x):
    import numpy as np

    return np.exp2(x.astype("float32")).astype("float32")


def _elementwise_exp2_prim_func(block_M, block_N, in_dtype, out_dtype):
    M = T.dynamic("m")
    N = T.dynamic("n")

    device_mesh_config = driver.get_sunmmio_device_mesh_config()

    zz_layout = make_zz_layout((M, N))
    placement = T.MeshShardingPolicy(y=0, x=1)

    @T.prim_func
    def elem_exp2(
        A: T.MeshTensor((M, N), placement, device_mesh_config, in_dtype, layout=zz_layout),
        B: T.MeshTensor((M, N), placement, device_mesh_config, out_dtype, layout=zz_layout),
    ):
        with T.Kernel() as _cid:
            sharded_M, sharded_N = A.local_shape

            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(A[bx * block_M, by * block_N], A_shared)
                    for i, j in T.Tiles([block_M, block_N]):
                        B_shared[i, j] = T.exp2(A_shared[i, j])
                    T.copy(B_shared, B[bx * block_M, by * block_N])

    return elem_exp2


@tilelang.jit(target="sunmmio", execution_backend="sunmmio_sunsim")
def elementwise_exp2_dynamic(block_M, block_N, in_dtype, out_dtype):
    return _elementwise_exp2_prim_func(block_M, block_N, in_dtype, out_dtype)


def main(M=1024, N=1024):
    import numpy as np
    import sunsim

    rng = np.random.default_rng(0)
    a = rng.uniform(-4.0, 4.0, size=(M, N)).astype(np.float32)

    kernel = elementwise_exp2_dynamic(
        block_M=32,
        block_N=32,
        in_dtype=T.float32,
        out_dtype=T.float32,
    )

    placement = [sunsim.S(0), sunsim.S(1)]
    layout = sunsim.Layout.zz(block_dims=(0, 1))
    b = sunsim.Output((M, N), np.float32, placement=placement, layout=layout)
    result = kernel(
        sunsim.Input(a, placement=placement, layout=layout),
        b,
        mesh=driver.get_sunmmio_device_mesh_config(),
        timeout=240.0,
    )

    np.testing.assert_allclose(b.data, ref_program(a), rtol=5e-4, atol=5e-4)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    args, _ = parser.parse_known_args()
    main(args.m, args.n)
