"""MEMORY software state uses u32 and preserves the i64 signal interface."""

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm

from testing.python.sunmmio.common.codegen_validation import codegen_sunmmio_suvm_mlir, validate_suvm_mlir_with_npuir_opt
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import collect_calls, collect_nodes, single_prim_func, strip_casts


@tilelang.jit(target="sunmmio")
def memory_wait_kernel(value=1, extra=0, group=False, world_size: int = 1):
    value_expr, extra_expr = tvm.tir.const(value, "uint64"), tvm.tir.const(extra, "uint64")

    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            if group:
                signals = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            else:
                signal = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            T.dist.wait_signal(signals[0] if group else signal, expected_delta=value_expr)
            T.dist.wait_signal(signals[0] if group else signal, expected_delta=extra_expr)

    return main


@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("value,extra", [(2**31 + 1, 0), (2**32 - 1, 0), (2**32 - 2, 1), (2**32 - 1, 1), (2**32, 0)])
def test_memory_software_representation_has_no_cumulative_limit(group, value, extra):
    func = memory_wait_kernel.get_tir(value, extra, group, world_size=2)
    result = lower_to_device_tir(func)
    waits = collect_calls(result.device_mod, "tl.dist_wait_signal_")
    assert all(wait.args[2].dtype == "uint64" for wait in waits)
    for wait in waits:
        state = strip_casts(wait.args[2])
        assert state.dtype == "uint32"
        assert state.buffer.scope() == ("shared.rsram" if group else "local.var")


@pytest.mark.parametrize("group", [False, True])
def test_memory_unsigned_extension_generates_extui(group, tmp_path):
    result = lower_to_device_tir(memory_wait_kernel.get_tir(2**31 + 7, group=group, world_size=2))
    func = single_prim_func(result.device_mod)

    def consume_wait(node):
        if isinstance(node, tvm.tir.Call) and node.op.same_as(tvm.ir.Op.get("tl.dist_wait_signal_")):
            return node.args[2]

    # Retain and consume the actual interface operand while isolating the
    # scalar/storage backend from communication.
    body = tvm.tir.stmt_functor.ir_transform(func.body, None, consume_wait, ["tir.Call"])
    # Remove the physical-resource request together with the communication.
    isolated = func.with_body(body).without_attr("tl.dist.signal_counts")
    source = codegen_sunmmio_suvm_mlir(tvm.IRModule({"memory_state": isolated}), log_ir=False)
    assert "arith.extui" in source and "i32 to i64" in source
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate", "--suvm-to-llvm-pipeline"))


@tilelang.jit(target="sunmmio")
def memory_completion_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            signals = T.dist.signals(2, kind=T.dist.SignalKind.SRAM_MEMORY, expect="manual")
            deltas = T.alloc_shared((16,), T.uint32)
            deltas[0] = T.uint32(2**31 + 7)
            deltas[1] = T.uint32(0)
            done = T.dist.completion(signals, expected_deltas=deltas[0:2])
            T.dist.wait_all(done)

    return main


def test_memory_completion_snapshot_and_update_use_unsigned_u32():
    result = lower_to_device_tir(memory_completion_kernel.get_tir(world_size=2), capture_passes="tl.LowerDistCommunication")
    initializer = collect_calls(result.pass_snapshot("tl.LowerDistCommunication").mod, "tl.dist_completion_init_")[0]
    for region in initializer.args[:2]:
        assert region.args[0].buffer.dtype == "uint32"
        assert region.args[0].buffer.scope() == "shared.rsram"
    wait_any = collect_calls(result.device_mod, "tl.dist_wait_any_")[0]
    assert wait_any.args[3].args[0].buffer.dtype == "uint32"
    assert int(wait_any.args[2]) == 2
    snapshot = wait_any.args[3].args[0].buffer
    stores = collect_nodes(result.device_mod, tvm.tir.BufferStore, lambda node: node.buffer.data.same_as(snapshot.data))
    assert stores and all(node.value.dtype == "uint32" for node in stores)
