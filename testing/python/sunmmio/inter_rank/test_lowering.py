"""Pass snapshots, compilation phases, and device kernel extraction in the lowering helper."""

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.utils.target import target_is_sunmmio
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from tvm import tir
from testing.python.sunmmio.inter_rank.tir_test_utils import (
    collect_op_names,
)


def test_lowering_captures_passes_and_extracts_device_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((32, 32), T.bfloat16),
        B: T.Tensor((32, 32), T.bfloat16),
    ):
        with T.Kernel():
            src = T.alloc_shared((32, 32), T.bfloat16, scope="shared.rsram")
            dst = T.alloc_shared((32, 32), T.bfloat16, scope="shared.rsram")

            T.copy(A, src)
            T.comm.broadcast(src, dst, (0, 0), direction="h")
            T.copy(dst, B)

    result = lower_to_device_tir(
        main,
        capture_passes=("tl.LowerTileOp", "tl.InjectSunmmioSync"),
    )
    device_mod = result.device_mod
    device_funcs = [(gvar, func) for gvar, func in device_mod.functions.items() if isinstance(func, tir.PrimFunc)]

    assert len(device_funcs) == 1
    gvar, device_func = device_funcs[0]
    assert gvar.name_hint == "main_kernel"
    assert str(device_func.attrs["global_symbol"]) == "main_kernel"
    assert bool(device_func.attrs["tir.is_global_func"])
    assert target_is_sunmmio(device_func.attrs["target"])

    lower_tile_op_snapshot = result.pass_snapshot("tl.LowerTileOp")
    assert lower_tile_op_snapshot.execution.phase == "LowerAndLegalize"
    lower_tile_op_names = collect_op_names(lower_tile_op_snapshot.mod)
    assert "tl.tileop.comm_broadcast" not in lower_tile_op_names
    assert lower_tile_op_names.count("tl.broadcast_") == 1

    inject_sync_snapshot = result.pass_snapshot("tl.InjectSunmmioSync")
    assert inject_sync_snapshot.execution.phase == "OptimizeForTarget"
    inject_sync_op_names = collect_op_names(inject_sync_snapshot.mod)
    assert "tl.barrier_arrive_and_wait" in inject_sync_op_names
    assert "tl.sunmmio_sync" in inject_sync_op_names
    assert "tl.sync_token_id" not in inject_sync_op_names
    assert "tl.wait_token" not in inject_sync_op_names

    device_op_names = collect_op_names(device_func)
    assert device_op_names.count("tl.broadcast_") == 1
    assert "tl.barrier_arrive_and_wait" in device_op_names
    assert "tl.sunmmio_sync" in device_op_names
    assert "tl.sync_token_id" not in device_op_names
    assert "tl.wait_token" not in device_op_names


if __name__ == "__main__":
    tilelang.testing.main()
