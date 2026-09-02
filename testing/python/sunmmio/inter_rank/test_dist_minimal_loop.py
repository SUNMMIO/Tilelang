"""Rank 间通信 put + signal + wait 最小闭环测试。"""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


@tilelang.jit(target="sunmmio")
def minimal_put_kernel_factory(M, N, world_size: int = 1):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        A: T.MeshTensor((M, N), placement, T.bfloat16),  # type: ignore
        B: T.MeshTensor((M, N), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            local_M, local_N = A.local_shape
            src = T.alloc_shared((local_M, local_N), T.bfloat16)
            dst = T.alloc_shared((local_M, local_N), T.bfloat16)
            signal = T.dist.signal()

            T.copy(A, src)
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal)
            T.dist.wait_signal(signal, dst=dst)
            T.dist.wait()
            T.copy(dst, B)

    return main


@tilelang.jit(target="sunmmio")
def too_many_signals_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal_0 = T.dist.signal()
            signal_1 = T.dist.signal()
            signal_2 = T.dist.signal()
            signal_3 = T.dist.signal()
            signal_4 = T.dist.signal()
            signal_5 = T.dist.signal()
            signal_6 = T.dist.signal()
            signal_7 = T.dist.signal()
            signal_8 = T.dist.signal()
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_0)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_1)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_2)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_3)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_4)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_5)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_6)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_7)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal_8)

    return main


@tilelang.jit(target="sunmmio")
def explicit_memory_signal_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY)
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal)
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def reused_signal_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32, 32), T.bfloat16)
            dst = T.alloc_shared((32, 32), T.bfloat16)
            signal = T.dist.signal()
            peer_rank = (rank_id + 1) % world_size
            for _step in T.serial(2):
                T.dist.put(src, dst, dst_rank=peer_rank, signal=signal)
                T.dist.wait_signal(signal, dst=dst)
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def invalid_payload_scope_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32, 32), T.bfloat16, scope="shared.asram")
            dst = T.alloc_shared((32, 32), T.bfloat16, scope="shared.asram")
            signal = T.dist.signal()
            peer_rank = (rank_id + 1) % T.dist.world_size()
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal)
            T.dist.wait_signal(signal, dst=dst)
            T.dist.wait()

    return main


@tilelang.jit(target="sunmmio")
def signal_in_loop_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            for _step in T.serial(2):
                T.dist.signal()

    return main


@tilelang.jit(target="sunmmio")
def two_puts_two_waits_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32, 32), T.bfloat16)
            dst = T.alloc_shared((32, 32), T.bfloat16)
            signal = T.dist.signal()
            peer_rank = (rank_id + 1) % world_size
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal)
            T.dist.put(src, dst, dst_rank=peer_rank, signal=signal)
            T.dist.wait_signal(signal, dst=dst)
            T.dist.wait_signal(signal, dst=dst)
            T.dist.wait()

    return main


def _collect_op_names(func_or_mod):
    names = []
    funcs = func_or_mod.functions.values() if isinstance(func_or_mod, tvm.IRModule) else (func_or_mod,)

    def visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            names.append(node.op.name)

    for func in funcs:
        if isinstance(func, tvm.tir.PrimFunc):
            tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return names


def _collect_op_arg_counts(func_or_mod, op_name):
    counts = []
    funcs = func_or_mod.functions.values() if isinstance(func_or_mod, tvm.IRModule) else (func_or_mod,)

    def visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == op_name:
            counts.append(len(node.args))

    for func in funcs:
        if isinstance(func, tvm.tir.PrimFunc):
            tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return counts


def _single_device_func(device_mod):
    funcs = [func for func in device_mod.functions.values() if isinstance(func, tvm.tir.PrimFunc)]
    assert len(funcs) == 1
    return funcs[0]


def test_minimal_dist_frontend_emits_high_level_ops_and_signal_metadata():
    func = minimal_put_kernel_factory.get_tir(32, 32, world_size=4)
    op_names = _collect_op_names(func)

    assert "tl.dist.sram_signal_count" not in func.attrs
    assert op_names.count("tl.dist_signal_decl") == 1
    assert "tl.dist_signal" not in op_names
    assert op_names.count("tl.tileop.dist_put") == 1
    assert op_names.count("tl.tileop.dist_wait_signal") == 1
    assert op_names.count("tl.dist_wait_send") == 1
    assert "tl.dist_put_" not in op_names
    frontend_script = func.script()
    assert 'T.dist_signal_decl("auto", 0)' in frontend_script
    assert "T.dist_put" in frontend_script
    assert "local.var" not in frontend_script


def test_minimal_dist_pipeline_lowers_to_device_leaf_ops():
    func = minimal_put_kernel_factory.get_tir(32, 32, world_size=4)
    result = lower_to_device_tir(
        func,
        capture_passes=(
            "tl.PlanDistSignals",
            "tl.LowerDistRouting",
            "tl.LowerDistCommunication",
            "tl.LowerTileOp",
            "tl.InjectDistSync",
        ),
    )

    planned_signal = result.pass_snapshot("tl.PlanDistSignals").mod
    assert 'T.dist_signal("sram_flagreg_inc", 0)' in planned_signal.script()
    assert int(planned_signal["main"].attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1

    routed_names = _collect_op_names(result.pass_snapshot("tl.LowerDistRouting").mod)
    assert "tl.tileop.dist_put" not in routed_names
    assert routed_names.count("tl.tileop.dist_peer_put") == 1
    assert "tl.tileop.comm_put" not in routed_names

    planned_names = _collect_op_names(result.pass_snapshot("tl.LowerDistCommunication").mod)
    assert "tl.tileop.dist_put" not in planned_names
    assert "tl.dist_signal_decl" not in planned_names
    assert "tl.dist_put_" in planned_names
    assert "tl.dist_wait_signal_" in planned_names
    assert "tl.dist_expect_" in planned_names
    planned_script = result.pass_snapshot("tl.LowerDistCommunication").mod.script()
    assert "tl.dist_signal" not in planned_names
    assert "signal_expect" in planned_script
    assert "signal_generation" in planned_script
    assert "signal_expect_1[0] =" not in planned_script
    assert "signal_generation_1[0] =" not in planned_script
    assert _collect_op_arg_counts(result.pass_snapshot("tl.LowerDistCommunication").mod, "tl.dist_put_") == [6]
    assert _collect_op_arg_counts(result.pass_snapshot("tl.LowerDistCommunication").mod, "tl.dist_wait_signal_") == [4]

    lowered_names = _collect_op_names(result.pass_snapshot("tl.LowerTileOp").mod)
    assert "tl.tileop.dist_put" not in lowered_names
    assert "tl.dist_signal_decl" not in lowered_names
    assert "tl.dist_signal" not in lowered_names
    assert "tl.dist_put_" in lowered_names
    assert "tl.dist_wait_signal_" in lowered_names
    assert "tl.dist_wait_send" in lowered_names
    lowered_script = result.pass_snapshot("tl.LowerTileOp").mod.script()
    assert "signal_expect" in lowered_script
    assert "signal_generation" in lowered_script
    assert "signal_expect_1[0] =" not in lowered_script
    assert _collect_op_arg_counts(result.pass_snapshot("tl.LowerTileOp").mod, "tl.dist_put_") == [6]
    assert _collect_op_arg_counts(result.pass_snapshot("tl.LowerTileOp").mod, "tl.dist_wait_signal_") == [4]

    injected_names = _collect_op_names(result.pass_snapshot("tl.InjectDistSync").mod)
    assert "tl.dist_put_" in injected_names
    assert "tl.dist_wait_signal_" in injected_names
    assert "tl.dist_wait_send" in injected_names
    injected_script = result.pass_snapshot("tl.InjectDistSync").mod.script()
    assert "signal_expect" in injected_script
    assert "T.dist_signal_decl" not in injected_script
    assert "T.dist_signal(" not in injected_script
    assert "signal_expect_1[0] =" in injected_script
    assert "signal_generation_1[0] =" in injected_script
    assert _collect_op_arg_counts(result.pass_snapshot("tl.InjectDistSync").mod, "tl.dist_put_") == [6]
    assert _collect_op_arg_counts(result.pass_snapshot("tl.InjectDistSync").mod, "tl.dist_wait_signal_") == [4]

    device_func = _single_device_func(result.device_mod)
    device_names = _collect_op_names(device_func)
    assert int(device_func.attrs["tl.dist.world_size"]) == 4
    assert "tl.dist.sram_signal_count" not in device_func.attrs
    assert int(device_func.attrs["tl.dist.signal_counts"]["sram_flagreg_inc"]) == 1
    assert device_names.count("tl.dist_signal_decl") == 0
    assert device_names.count("tl.dist_signal") == 0
    assert device_names.count("tl.dist_put_") == 1
    assert device_names.count("tl.dist_wait_signal_") == 1
    assert device_names.count("tl.dist_wait_send") == 1

    script = device_func.script()
    assert "T.dist_signal_decl(" not in script
    assert "T.dist_signal(" not in script
    assert "T.dist_put_(" in script
    assert "T.dist_wait_signal_(" in script
    assert "T.dist_wait_send()" in script
    assert "signal_expect" in script
    assert "signal_generation" in script
    assert "signal_expect_1[0] = signal_expect_1[0] + T.uint8(1)" in script
    assert "signal_generation_1[0] = signal_generation_1[0] + T.uint8(1)" in script
    assert "uint8" in script
    assert "T.wait_token(0)" in script
    assert script.index("T.wait_token(0)") < script.index("T.dist_put_(")


def test_explicit_memory_signal_reaches_stable_leaf_tir():
    func = explicit_memory_signal_kernel_factory.get_tir(world_size=4)
    device_func = _single_device_func(lower_to_device_tir(func).device_mod)
    script = device_func.script()

    assert "T.dist_put_(" in script
    assert ', "sram_memory", 0, signal_generation_1[0])' in script
    assert 'T.dist_wait_signal_("sram_memory", 0, signal_expect_1[0]' in script


def test_signal_rejects_more_than_eight_sram_flagregs():
    func = too_many_signals_kernel_factory.get_tir(world_size=4)
    with pytest.raises(tvm.error.InternalError, match="capacity exceeded"):
        lower_to_device_tir(func)


def test_signal_generation_is_reused_across_serial_loop():
    func = reused_signal_kernel_factory.get_tir(world_size=4)
    result = lower_to_device_tir(func)
    device_func = _single_device_func(result.device_mod)
    op_names = _collect_op_names(device_func)
    script = device_func.script()

    assert op_names.count("tl.dist_put_") == 1
    assert op_names.count("tl.dist_wait_signal_") == 1
    assert op_names.count("tl.dist_wait_send") == 1
    assert "for _step in range(2):" in script
    assert "signal_expect_1[0] = signal_expect_1[0] + T.uint8(1)" in script
    assert "signal_generation_1[0] = signal_generation_1[0] + T.uint8(1)" in script
    assert 'T.dist_wait_signal_("sram_flagreg_inc", 0, signal_expect_1[0]' in script


def test_put_advances_generation_and_wait_only_reads_it():
    func = two_puts_two_waits_kernel_factory.get_tir(world_size=4)
    device_func = _single_device_func(lower_to_device_tir(func).device_mod)
    script = device_func.script()

    assert script.count("signal_expect_1[0] = signal_expect_1[0] + T.uint8(1)") == 2
    assert script.count("signal_generation_1[0] = signal_generation_1[0] + T.uint8(1)") == 2
    assert script.count("T.dist_put_(") == 2
    assert script.count('T.dist_wait_signal_("sram_flagreg_inc", 0, signal_expect_1[0]') == 2

    lines = [line.strip() for line in script.splitlines()]
    advance_indices = [index for index, line in enumerate(lines) if line.startswith("signal_generation_1[0] =")]
    put_indices = [index for index, line in enumerate(lines) if line.startswith("T.dist_put_(")]
    wait_indices = [index for index, line in enumerate(lines) if line.startswith("T.dist_wait_signal_(")]
    assert advance_indices[0] < put_indices[0] < advance_indices[1] < put_indices[1]
    assert put_indices[1] < wait_indices[0] < wait_indices[1]


def test_signal_planning_rejects_unsupported_destination_scope():
    func = invalid_payload_scope_kernel_factory.get_tir(world_size=4)
    target = tvm.target.Target(determine_target("sunmmio", return_object=True))
    mod = tvm.IRModule({"main": func})
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with pytest.raises(tvm.error.InternalError, match="destination must use shared.rsram or global/DRAM"):
        tilelang.transform.PlanDistSignals()(mod)


def test_signal_rejects_loop_local_declaration():
    with pytest.raises(RuntimeError, match="outside loops and conditionals"):
        signal_in_loop_kernel_factory.get_tir(world_size=4)


def test_validate_dist_reports_communication_op_in_single_rank_kernel():
    func = minimal_put_kernel_factory.get_tir(32, 32, world_size=1)
    with pytest.raises(tvm.error.InternalError, match="world_size=1 disables Rank communication"):
        lower_to_device_tir(func)


def test_existing_non_dist_kernel_is_unchanged_by_dist_passes():
    @T.prim_func
    def main(A: T.Tensor((32,), T.float32)):
        with T.Kernel():
            A[0] = A[0] + 1

    target = tvm.target.Target("llvm")
    mod = tvm.IRModule({"main": main.with_attr("target", target)})
    for transform in (
        tilelang.transform.PlanDistSignals(),
        tilelang.transform.LowerDistRouting(),
        tilelang.transform.LowerDistCommunication(),
        tilelang.transform.InjectDistSync(),
    ):
        after = transform(mod)
        assert tvm.ir.structural_equal(after, mod)
