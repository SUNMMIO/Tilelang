import tilelang.language as T
from tilelang.layout import make_zz_layout
from tilelang.carver.arch import driver


def matmul_persistent(M, N, K, block_M, block_N, block_K, num_stages, dtype=T.bfloat16, accum_dtype=T.float32):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), device_mesh_config, dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), device_mesh_config, dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), device_mesh_config, accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A_shared_dist = T.alloc_shared((block_M, block_K * ncols), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            B_shared_dist = T.alloc_shared((block_K * nrows, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            # Each core iterates its own sharded tile grid with plain nested
            # loops. The MeshTensor sharding already distributes A/B/C across
            # the core mesh, so no persistent core-distribution loop is used.
            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_shared)
                    for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=num_stages):
                        # Stage each A/B tile into a shared buffer, then
                        # all-gather it across the core row / column.
                        # all_gather needs a Buffer source (not an indexed
                        # element) and an explicit concat axis.
                        T.copy(A[bx * block_M, k * block_K], A_shared)
                        T.comm.all_gather(A_shared, A_shared_dist, direction="horizontal", axis=-1)
                        T.copy(B[k * block_K, by * block_N], B_shared)
                        T.comm.all_gather(B_shared, B_shared_dist, direction="vertical", axis=0)
                        T.gemm(A_shared_dist, B_shared_dist, C_shared)

                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return main
