"""Backend-only tests for the native uint64 tir.For contract in device TIR."""

from dataclasses import replace

import pytest
from tilelang import tvm

from testing.python.sunmmio.codegen.test_dist_integer_control_flow_opt_validate import wide_window_kernel
from testing.python.sunmmio.common.codegen_validation import codegen_sunmmio_suvm_mlir, validate_suvm_mlir_with_npuir_opt
from testing.python.sunmmio.common.suvm_trace import trace_suvm_control_flow
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


def native_window(extent, step):
    # Keep the production kernel's ABI, logical IV use and signal operations.
    # The public DSL already encodes step in the body. Set the loop controls
    # directly here to exercise the backend's separate native For.step path.
    result = lower_to_device_tir(wide_window_kernel.get_tir(span=5, mode="offset", world_size=2))
    symbol, func = next(iter(result.device_mod.functions.items()))
    n = next(param for param in func.params if param.name == "n")

    def operand(value):
        return n if value == "runtime" else tvm.tir.const(value, "uint64")

    loops = []

    def set_controls(node):
        loops.append(node)
        return tvm.tir.For(
            node.loop_var, node.min, operand(extent), node.kind, node.body, node.thread_binding, node.annotations, step=operand(step)
        )

    body = tvm.tir.stmt_functor.ir_transform(func.body, None, set_controls, ["tir.For"])
    assert len(loops) == 1 and loops[0].loop_var.dtype == "uint64"
    return replace(result, device_mod=tvm.IRModule({symbol: func.with_body(body)}))


@pytest.mark.parametrize("step", [2, 1 << 63])
def test_native_step_restores_logical_iv(tmp_path, step):
    result = native_window(5, step)
    source = codegen_sunmmio_suvm_mlir(result.device_mod)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
    for n in (0, (1 << 63) - 1, (1 << 64) - 6):
        _, events = trace_suvm_control_flow(result, source, rank_id=0, n=n)
        assert [e[1] for e in events if e[0] == "wait"] == ([0, 2, 6] if step == 2 else [0])


@pytest.mark.parametrize(
    "extent,step,diagnostic",
    [
        ("runtime", 1, "trip count must fit"),
        (5, 0, "provably positive step"),
        (5, "runtime", "provably positive step"),
    ],
)
def test_native_loop_rejects_unproven_controls(extent, step, diagnostic):
    result = native_window(extent, step)
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        codegen_sunmmio_suvm_mlir(result.device_mod)


@pytest.mark.parametrize("extent", [(1 << 63) - 1, 1 << 63])
def test_native_trip_count_signed_index_limit(tmp_path, extent):
    result = native_window(extent, 1)
    if extent == 1 << 63:
        with pytest.raises(tvm.error.InternalError, match="trip count must fit"):
            codegen_sunmmio_suvm_mlir(result.device_mod)
        return
    source = codegen_sunmmio_suvm_mlir(result.device_mod)
    validate_suvm_mlir_with_npuir_opt(source, tmp_path, opt_args=("--verify-each", "--suvm-device-validate"))
