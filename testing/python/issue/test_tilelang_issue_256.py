import numpy as np
import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.carver.arch import driver
from tilelang.layout import make_row_major


ml_dtypes = pytest.importorskip("ml_dtypes")
sunsim = pytest.importorskip("sunsim")


M = 128
N = 32


@tilelang.jit(target="sunmmio", execution_backend="sunmmio_sunsim")
def allgather_dram_to_rsram():
    input_policy = T.MeshShardingPolicy(x=0, replicate=T.MeshReplicationType.COLUMN)
    output_policy = T.MeshShardingPolicy(replicate=T.MeshReplicationType.ALL)
    input_layout = make_row_major((M, N))
    output_layout = make_row_major((M, N))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, N), input_policy, T.bfloat16, layout=input_layout),
        B: T.MeshTensor((M, N), output_policy, T.bfloat16, layout=output_layout),
    ):
        with T.Kernel():
            A_gathered = T.alloc_shared((M, N), T.bfloat16, scope="shared.rsram")
            T.annotate_layout({A_gathered: output_layout})
            T.comm.all_gather(A, A_gathered, direction="h", axis=0)
            T.copy(A_gathered, B)

    return main


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_sunsim
@pytest.mark.sunmmio_closed_runtime
def test_allgather_dram_to_rsram():
    source = np.arange(M * N, dtype=np.float32).reshape(M, N).astype(ml_dtypes.bfloat16)
    result = sunsim.Output(
        (M, N),
        ml_dtypes.bfloat16,
        placement=[sunsim.R(), sunsim.R()],
        layout=sunsim.Layout.nd(),
    )

    kernel = allgather_dram_to_rsram()
    run = kernel(
        sunsim.Input(
            source,
            placement=[sunsim.R(), sunsim.S(0)],
            layout=sunsim.Layout.nd(),
        ),
        result,
        mesh=driver.get_sunmmio_device_mesh_config(),
        timeout=240.0,
    )

    assert run.exit_code == 0
    np.testing.assert_array_equal(result.data, source)

    kernel_source = kernel.get_kernel_source()
    copy_pos = kernel_source.index("suvm.copy_async")
    wait_pos = kernel_source.index("suvm.wait_token", copy_pos)
    mcast_pos = kernel_source.index("suvm.mcast_tok", wait_pos)
    assert copy_pos < wait_pos < mcast_pos


if __name__ == "__main__":
    tilelang.testing.main()
