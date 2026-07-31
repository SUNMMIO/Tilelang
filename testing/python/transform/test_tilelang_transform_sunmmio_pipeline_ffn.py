import pytest
import tilelang as tl
from tilelang import tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir

from testing.python.transform.sunmmio_mesh_kernel_new_syntax_reference import mesh_ffn_new
from testing.python.transform.test_tilelang_transform_sunmmio_pipeline_strict import (
    lower_and_legalize_sunmmio_pipeline_test,
)


def _pipeline_loops(stmt):
    loops = []

    def visit(node):
        if node is None:
            return
        if isinstance(node, tir.For):
            annotations = node.annotations
            if annotations and (
                "num_stages" in annotations
                or ("prologue_orders" in annotations and "body_orders" in annotations)
                or "tl.sunmmio.pipeline.requested" in annotations
            ):
                loops.append(node)
            visit(node.body)
        elif isinstance(node, tir.BlockRealize):
            visit(node.block.body)
        elif isinstance(node, tir.Block):
            visit(node.body)
        elif isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child)
        elif isinstance(node, tir.IfThenElse):
            visit(node.then_case)
            visit(node.else_case)
        elif isinstance(node, (tir.LetStmt, tir.AttrStmt)):
            visit(node.body)

    visit(stmt)
    return loops


def _lower_ffn(num_stages):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        func = mesh_ffn_new(num_stages=num_stages).with_attr("global_symbol", "main")
        mod = tvm.IRModule.from_expr(func)
        mod = lower_and_legalize_sunmmio_pipeline_test(mod, target)
        return tl.transform.IfStmtBinding()(mod)


def _assert_two_collective_fallback_loops(mod):
    loops = _pipeline_loops(mod["main"].body)
    assert len(loops) == 2
    for loop in loops:
        assert bool(loop.annotations["tl.sunmmio.pipeline.requested"])
        assert not bool(loop.annotations["tl.sunmmio.pipeline.applied"])
        assert str(loop.annotations["tl.sunmmio.pipeline.fallback_stage"]) == "planning"
        assert str(loop.annotations["tl.sunmmio.pipeline.fallback_reason"]) == "repeated_collective_destination"


def _assert_striped_broadcast_writers_are_independent(loop, writer_ids=(2, 5)):
    assert bool(loop.annotations["tl.sunmmio.pipeline.requested"])
    assert bool(loop.annotations["tl.sunmmio.pipeline.applied"])

    scheduled_ids = set()
    for name in ("prologue_orders", "body_orders", "epilogue_orders"):
        scheduled_ids.update(int(str(order).split("-")[1]) for order in loop.annotations[name])
    assert scheduled_ids == set(range(7))

    writer_phases = loop.annotations["runtime_bank_writer_phases"]
    grouped_phase_maps = []
    for phases in writer_phases.values():
        by_id = {int(command_id): int(phase) for command_id, phase in phases.items()}
        if all(writer_id in by_id for writer_id in writer_ids):
            grouped_phase_maps.append(by_id)
    assert len(grouped_phase_maps) == 1
    assert len({grouped_phase_maps[0][writer_id] for writer_id in writer_ids}) == 2


def _assert_all_gather_encounter_order(loop, collective_ids=(2, 3, 5)):
    orders = []
    for name in ("prologue_orders", "body_orders", "epilogue_orders"):
        orders.extend(tuple(map(int, str(order).split("-"))) for order in loop.annotations[name])

    collective_orders = [order for order in orders if order[1] in collective_ids]
    iterations = sorted({iteration for iteration, _ in collective_orders})
    assert collective_orders == [(iteration, command_id) for iteration in iterations for command_id in collective_ids]


def test_ffn_serial_has_two_unplanned_projection_loops():
    mod = _lower_ffn(num_stages=0)
    script = mod.script(show_meta=True)
    assert '"body_orders"' not in script
    assert script.count("T.mma_sunmmio(") >= 2


def test_ffn_greedy_repeated_collective_destination_falls_back_atomically():
    mod = _lower_ffn(num_stages=3)
    planned = tl.transform.SunmmioPipelinePlanning(debug=False)(mod)
    _assert_two_collective_fallback_loops(planned)

    injected = tl.transform.InjectSunmmioPipeline()(planned)
    script = injected.script(show_meta=True)
    assert '"body_orders"' not in script
    assert "mid_shared_ping" not in script
    assert "mid_shared_pong" not in script
    assert "down_shared_ping" not in script
    assert "down_shared_pong" not in script
    assert "WUp" in script
    assert "WDown" in script


@pytest.mark.parametrize("num_stages", (2, 3))
def test_ffn_ilp_striped_broadcast_writers_select_independent_banks_then_inject(
    num_stages,
):
    mod = _lower_ffn(num_stages=num_stages)
    planned = tl.transform.SunmmioPipelinePlanningILP(debug=False)(mod)
    loops = _pipeline_loops(planned["main"].body)
    assert len(loops) == 2
    for loop in loops:
        _assert_striped_broadcast_writers_are_independent(loop)
        _assert_all_gather_encounter_order(loop)
        if num_stages == 2:
            banked = {buffer.name for buffer in loop.annotations["runtime_banked_buffers"]}
            multiversion = {buffer.name for buffer in loop.annotations["runtime_multiversion_buffers"]}
            assert banked.isdisjoint(multiversion)

    injected = tl.transform.InjectSunmmioPipelineILP()(planned)
    script = injected.script(show_meta=True)
    assert '"tl.sunmmio.pipeline.fallback_reason"' not in script
    # Planning is atomic, but injection preserves both lowered broadcast leaves.
    broadcasts = []
    tir.stmt_functor.post_order_visit(
        injected["main"].body,
        lambda node: broadcasts.append(node)
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.broadcast_"
        else None,
    )
    assert len(broadcasts) >= 4
