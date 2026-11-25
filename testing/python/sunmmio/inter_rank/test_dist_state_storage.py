"""Physical storage and signal-boundary types for compiler state arrays."""

import os

import pytest
import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm

from testing.python.sunmmio.common.codegen_validation import (
    codegen_sunmmio_suvm_mlir,
    validate_suvm_mlir_with_npuir_opt,
    write_sunmmio_codegen_logs,
)
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import collect_calls, collect_nodes, single_prim_func

tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ.setdefault("SUNMMIO_TEST_LOG_IR", "0")


@tilelang.jit(target="sunmmio")
def state_array_kernel(count=2, delta=1, dense=False, kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signals = T.dist.signals(count, kind=kind, expect="manual")
            deltas = T.alloc_shared((count,), T.int32)
            T.fill(deltas, delta)
            if dense:
                if rank_id == 0:
                    T.dist.put_signal(signals[0], 1)
                    T.dist.put_signal(signals[0], 2)
                T.dist.submit()
            else:
                T.dist.put_signal(signals[0], (rank_id + 1) % world_size, submit=True)
            done = T.dist.completion(signals, expected_deltas=deltas)
            T.dist.wait_all(done)
            T.dist.wait_signal(signals[rank_id % count], expected_delta=0)

    return main


@pytest.mark.parametrize("count", [1, 16, 17, 40])
def test_state_arrays_use_rsram_and_padded_layout(count):
    # VALUE capacity is checked separately; use only the storage lowering for
    # the longer shape through an automatically planned group below.
    if count > 32:
        func = state_shape_kernel.get_tir(count, world_size=2)
    else:
        func = state_array_kernel.get_tir(count, world_size=2)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCommunication")
    lowered = single_prim_func(result.pass_snapshot("tl.LowerDistCommunication").mod)
    wait_any = collect_calls(result.device_mod, "tl.dist_wait_any_")[0]
    for i, arg in enumerate(wait_any.args[3:5]):
        buffer = arg.args[0].buffer
        assert buffer.scope() == "shared.rsram"
        assert buffer.dtype == ("uint32" if count > 32 and i == 0 else "int32")
        assert tuple(int(x) for x in buffer.shape) == (count,)
        assert int(arg.args[2]) == count
        device = single_prim_func(result.device_mod)
        layout = next(layout for candidate, layout in device.attrs["layout_map"].items() if candidate.data.same_as(buffer.data))
        shape = tvm.ffi.get_global_func("tl.CuteLayout_mode_shape")(layout)
        assert [int(x) for x in shape] == [((count + 15) // 16) * 16]
    assert collect_calls(lowered, "tl.dist_completion_init_")


@tilelang.jit(target="sunmmio")
def state_shape_kernel(count=40, delta=1, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signals = T.dist.signals(count, kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            deltas = T.alloc_shared((count,), T.uint32)
            T.fill(deltas, delta)
            done = T.dist.completion(signals, expected_deltas=deltas)
            T.dist.wait_all(done)

    return main


@pytest.mark.parametrize("dense", [False, True])
def test_state_generation_cast_follows_update(dense):
    result = lower_to_device_tir(state_array_kernel.get_tir(dense=dense, world_size=3 if dense else 2))
    device = single_prim_func(result.device_mod)
    send = collect_calls(device, "tl.dist_signal_put_")[0]
    value = send.args[3]
    assert isinstance(value, tvm.tir.Cast) and value.dtype == "uint8"
    assert value.value.buffer.dtype == "int32"
    assert value.value.buffer.scope() == "shared.rsram"
    stores = collect_nodes(device, tvm.tir.BufferStore, lambda store: store.buffer.data.same_as(value.value.buffer.data))
    assert any(isinstance(store.value, tvm.tir.Add) for store in stores)
    waits = collect_calls(device, "tl.dist_wait_signal_")
    assert waits and all(wait.args[2].dtype == "uint8" for wait in waits)
    assert not collect_calls(device, "tl.dist_completion_init_")
    assert any(alloc.buffer_var.type_annotation.storage_scope == "local.var" for alloc in collect_nodes(device, tvm.tir.Allocate))


@pytest.mark.parametrize("delta", [255, 256])
def test_rsram_expected_retains_flag_range(delta):
    func = state_array_kernel.get_tir(delta=delta, world_size=2)
    if delta == 256:
        with pytest.raises(tvm.error.InternalError, match="cumulative expected"):
            lower_to_device_tir(func)
    else:
        lower_to_device_tir(func)


@pytest.mark.parametrize("kind", [T.dist.SignalKind.SRAM_FLAGREG_VALUE, T.dist.SignalKind.SRAM_MEMORY])
def test_generated_state_access_validates_without_dist_handlers(tmp_path, kind):
    kernel = state_array_kernel.get_tir(kind=kind, world_size=2)
    write_sunmmio_codegen_logs(case_name=tmp_path.name, kernel=kernel)
    result = lower_to_device_tir(kernel)

    # Exercise actual compiler-created state accesses. Replacing communication
    # only isolates the existing scalar/tile backend; it does not test dist.
    def remove_dist(node):
        if isinstance(node, tvm.tir.Call) and node.op.name.startswith("tl.dist_"):
            return tvm.tir.const(0, node.dtype if node.dtype != "handle" else "int32")

    func = single_prim_func(result.device_mod)
    body = tvm.tir.stmt_functor.ir_transform(func.body, None, remove_dist, ["tir.Call"])
    # The isolated storage program has no physical signal resources.
    mod = tvm.IRModule({"state_access": func.with_body(body).without_attr("tl.dist.signal_counts")})
    write_sunmmio_codegen_logs(case_name=tmp_path.name, tir_mod=mod)
    source = codegen_sunmmio_suvm_mlir(mod, log_case_name=tmp_path.name)
    assert "suvm.tile.pick" in source and "suvm.tile.set" in source
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate", "--suvm-to-llvm-pipeline"))


@tilelang.jit(target="sunmmio")
def auto_state_kernel(kind=None, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            source = T.alloc_shared((32,), T.bfloat16)
            destination = T.alloc_shared((32,), T.bfloat16)
            signals = T.dist.signals(2, kind=kind)
            T.dist.put(source, destination, (rank_id + 1) % world_size, signal=signals[0], submit=True)
            done = T.dist.completion(signals)
            T.dist.wait_all(done)

    return main


@pytest.mark.parametrize("kind", [None, T.dist.SignalKind.SRAM_MEMORY])
def test_auto_completion_delta_array_uses_existing_rsram_storage(kind):
    result = lower_to_device_tir(auto_state_kernel.get_tir(kind=kind, world_size=2), capture_passes="tl.LowerDistCommunication")
    mod = result.pass_snapshot("tl.LowerDistCommunication").mod
    initializer = collect_calls(mod, "tl.dist_completion_init_")[0]
    for delta in initializer.args[5:]:
        assert delta.buffer.scope() == "shared.rsram"
        assert delta.buffer.dtype == ("uint32" if kind else "int32")
    assert not bool(initializer.args[4])
    device = single_prim_func(result.device_mod)
    snapshot = collect_calls(device, "tl.dist_wait_any_")[0].args[3].args[0].buffer
    expected = collect_calls(mod, "tl.dist_expect_")[0].args[0].buffer
    assert not snapshot.data.same_as(expected.data)


if __name__ == "__main__":
    tilelang.testing.main()
