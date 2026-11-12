"""Generic dist-region validation and mandatory complete-tail checks."""

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.layout import make_aligned_row_major, make_row_major, make_zz_layout
from tilelang.language.utils import region
from tilelang.utils import target as target_utils

from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir
from testing.python.sunmmio.inter_rank.tir_test_utils import collect_calls


@tilelang.jit(target="sunmmio")
def region_kernel(length=96, source_length=None, padded=False, offset=0, dynamic=False, scope="shared.rsram", world_size: int = 1):
    src_length = length if source_length is None else source_length
    src_shape, dst_shape = (src_length,), (length,)
    src_layout = make_aligned_row_major(src_shape, T.int32, align_bytes=128) if padded else make_row_major(src_shape)
    dst_layout = make_aligned_row_major(dst_shape, T.int32, align_bytes=128) if padded else make_row_major(dst_shape)

    @T.prim_func
    def main(
        src: T.MeshTensor(src_shape, T.placement.replicated(), T.int32, layout=src_layout),
        dst: T.MeshTensor(dst_shape, T.placement.replicated(), T.int32, layout=dst_layout),
        rank_id: T.dist.RankId,
        index: T.int32,
    ):
        with T.Kernel():
            local_src = T.alloc_shared(src_shape, T.int32)
            local_dst = T.alloc_shared(dst_shape, T.int32)
            T.annotate_layout({src: src_layout, dst: dst_layout, local_src: src_layout, local_dst: dst_layout})
            ready = T.dist.signal()
            begin = index * 32 if dynamic else offset
            if scope == "shared.rsram":
                T.dist.put(local_src[begin : begin + 32], local_dst[0:32], (rank_id + 1) % world_size, signal=ready, submit=True)
            elif scope == "dram_to_sram":
                T.dist.put(src[begin : begin + 32], local_dst[0:32], (rank_id + 1) % world_size, signal=ready, submit=True)
            elif scope == "sram_to_dram":
                T.dist.put(local_src[begin : begin + 32], dst[0:32], (rank_id + 1) % world_size, signal=ready, submit=True)
            else:
                T.dist.put(src[begin : begin + 32], dst[0:32], (rank_id + 1) % world_size, signal=ready, submit=True)
            T.dist.wait_signal(ready)

    return main


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("source_length,length,valid", [(96, 96, True), (100, 96, False), (96, 100, False)])
def test_dist_complete_tail_check_is_independent_of_optional_validation(monkeypatch, enabled, source_length, length, valid):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", enabled)
    func = region_kernel.get_tir(length, source_length, world_size=2)
    if valid:
        assert collect_calls(lower_to_device_tir(func).device_mod, "tl.dist_put_")
    else:
        with pytest.raises(tvm.error.InternalError, match="partial last tile"):
            lower_to_device_tir(func)


@pytest.mark.parametrize("offset,dynamic", [(0, False), (32, False), (0, True)])
def test_dist_padded_layout_uses_covered_extent_without_expanding_payload(monkeypatch, offset, dynamic):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", True)
    result = lower_to_device_tir(region_kernel.get_tir(100, padded=True, offset=offset, dynamic=dynamic, world_size=2))
    put = collect_calls(result.device_mod, "tl.dist_put_")[0]
    for payload in put.args[:2]:
        if isinstance(payload, tvm.tir.BufferLoad):
            assert int(payload.indices[0].lanes) == 32
        else:
            assert int(payload.args[2]) == 32


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("offset,error", [(1, "align to region extent"), (96, "within buffer shape")])
def test_dist_generic_checks_follow_existing_switch(monkeypatch, enabled, offset, error):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", enabled)
    func = region_kernel.get_tir(offset=offset, world_size=2)
    if enabled:
        with pytest.raises(tvm.error.InternalError, match=error):
            lower_to_device_tir(func)
    else:
        lower_to_device_tir(func)


@pytest.mark.parametrize("scope", ["shared.rsram", "dram_to_sram", "sram_to_dram", "dram"])
def test_dist_region_validation_accepts_existing_memory_paths(monkeypatch, scope):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", True)
    lower_to_device_tir(region_kernel.get_tir(scope=scope, world_size=2))


@pytest.mark.parametrize("operation", ["dist_put", "dist_peer_put", "dist_routed_peer_put"])
@pytest.mark.parametrize("operand", ["src", "dst"])
@pytest.mark.parametrize("padded", [False, True])
def test_generic_dist_dispatch_checks_all_region_pairs(operation, operand, padded):
    shape = (100,) if padded else (96,)
    layout = make_aligned_row_major(shape, T.int32, align_bytes=128) if padded else make_row_major(shape)
    src = tvm.tir.decl_buffer(shape, "int32", name="source", scope="shared.rsram")
    dst = tvm.tir.decl_buffer(shape, "int32", name="destination", scope="shared.rsram")
    valid = [region(src[0], "r", 32), region(dst[0], "w", 32)]
    invalid = [region(src[96 if operand == "src" else 0], "r", 32), region(dst[96 if operand == "dst" else 0], "w", 32)]
    zero = tvm.tir.const(0, "int32")
    if operation == "dist_routed_peer_put":
        args = [zero, zero, zero, *valid, *invalid]
    else:
        args = [*invalid, zero, zero, zero]
        if operation == "dist_put":
            args.append(zero)
    call = tvm.tir.call_intrin("handle", tvm.ir.Op.get("tl.tileop." + operation), *args)
    block = tvm.tir.Block([], [], [], "regions", tvm.tir.Evaluate(call), annotations={"layout_map": {src: layout, dst: layout}})
    func = tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, block)).with_attr(
        "target", tvm.target.Target(target_utils.SUNMMIO_TARGET_DESC)
    )
    with pytest.raises(tvm.error.InternalError, match=operand + ".*within buffer shape"):
        tilelang.transform.ValidateTileViewRegions()(tvm.IRModule({"main": func}))


@tilelang.jit(target="sunmmio")
def noncontiguous_region_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((3, 96, 96), T.int32)
            dst = T.alloc_shared((32, 32), T.int32)
            T.annotate_layout({src: make_row_major((3, 96, 96)), dst: make_row_major((32, 32))})
            ready = T.dist.signal()
            T.dist.put(src[1:2, 32:64, 32:64], dst, (rank_id + 1) % world_size, signal=ready, submit=True)
            T.dist.wait_signal(ready)

    return main


def test_dist_accepts_noncontiguous_region_and_non_tiled_unit_axis(monkeypatch):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", True)
    lower_to_device_tir(noncontiguous_region_kernel.get_tir(world_size=2))


@tilelang.jit(target="sunmmio")
def blocked_region_kernel(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((33, 64), T.bfloat16)
            dst = T.alloc_shared((64, 64), T.bfloat16)
            T.annotate_layout({src: make_zz_layout((33, 64)), dst: make_zz_layout((64, 64))})
            ready = T.dist.signal()
            T.dist.put(src[0:32, 0:32], dst[0:32, 0:32], (rank_id + 1) % world_size, signal=ready, submit=True)
            T.dist.wait_signal(ready)

    return main


def test_dist_blocked_padding_does_not_require_logical_shape_divisibility(monkeypatch):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", True)
    lower_to_device_tir(blocked_region_kernel.get_tir(world_size=2))


def test_dist_frontend_clipping_is_preserved_with_padded_layout(monkeypatch):
    monkeypatch.setattr(target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", True)
    with pytest.warns(UserWarning, match="will be clipped"):
        func = region_kernel.get_tir(100, padded=True, offset=96, world_size=2)
    result = lower_to_device_tir(func)
    put = collect_calls(result.device_mod, "tl.dist_put_")[0]
    # Preserve the frontend's clipped four elements, not the padded capacity.
    assert int(put.args[0].indices[0].lanes) == 4
