import argparse

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.layout import make_zz_layout


def ref_program(x, y):
    return x.astype("float32") + y.astype("float32")


def _elementwise_add_prim_func(M, N, block_M, block_N, in_dtype, out_dtype):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()

    zz_layout = make_zz_layout((M, N))
    placement = T.MeshShardingPolicy(y=0, x=1)

    @T.prim_func
    def elem_add(
        A: T.MeshTensor((M, N), placement, device_mesh_config, in_dtype, layout=zz_layout),
        B: T.MeshTensor((M, N), placement, device_mesh_config, in_dtype, layout=zz_layout),
        C: T.MeshTensor((M, N), placement, device_mesh_config, out_dtype, layout=zz_layout),
    ):
        with T.Kernel() as _cid:
            sharded_M, sharded_N = A.local_shape

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


@tilelang.jit(target="sunmmio", execution_backend="sunmmio_sunsim")
def elementwise_add(M, N, block_M, block_N, in_dtype, out_dtype):
    return _elementwise_add_prim_func(M, N, block_M, block_N, in_dtype, out_dtype)


def main(M=1024, N=1024):
    import ml_dtypes
    import numpy as np
    import sunsim

    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, N)).astype(np.float32).astype(ml_dtypes.bfloat16)
    b = rng.standard_normal((M, N)).astype(np.float32).astype(ml_dtypes.bfloat16)

    kernel = elementwise_add(
        M,
        N,
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )

    placement = [sunsim.S(0), sunsim.S(1)]
    layout = sunsim.Layout.zz(block_dims=(0, 1))
    c = sunsim.Output((M, N), np.float32, placement=placement, layout=layout)
    result = kernel(
        sunsim.Input(a, placement=placement, layout=layout),
        sunsim.Input(b, placement=placement, layout=layout),
        c,
        mesh=driver.get_sunmmio_device_mesh_config(),
        timeout=240.0,
    )

    np.testing.assert_allclose(c.data, ref_program(a, b), rtol=1e-2, atol=1e-2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    args, _ = parser.parse_known_args()
    main(args.m, args.n)
