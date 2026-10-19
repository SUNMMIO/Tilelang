import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_row_major, make_zn_layout, make_zz_layout
from tilelang.utils.target import SUNMMIO_TARGET_DESC


DTYPE = "float16"


def _tile_region(buffer_load, access_type, *extents):
    access_code = {"r": 1, "w": 2, "rw": 3}[access_type]
    return tvm.tir.call_intrin(
        "handle",
        tvm.tir.op.Op.get("tl.tileop.region"),
        buffer_load,
        access_code,
        *extents,
    )


@pytest.fixture(autouse=True)
def disable_tilelang_cache():
    cache_was_enabled = tilelang.env.is_cache_enabled()
    tilelang.env.disable_cache()
    try:
        yield
    finally:
        if cache_was_enabled:
            tilelang.env.enable_cache()
        else:
            tilelang.env.disable_cache()


@pytest.fixture
def _strict_region_validation():
    import tilelang.utils.target as target_utils

    previous = target_utils.ENABLE_SUNMMIO_REGION_VALIDATION
    target_utils.set_sunmmio_region_validation(True)
    try:
        yield
    finally:
        target_utils.set_sunmmio_region_validation(previous)


def _make_copy_kernel(copy_case):
    if copy_case in {
        "dynamic_extent_warns_and_passes",
        "dynamic_row_major_inner_mode_warns_and_passes",
    }:
        shape = (T.dynamic("m"), 96)
        layout = make_row_major(shape)
    elif copy_case in {
        "row_major_3d_outer_slab_valid",
        "row_major_3d_outer_slab_unclipped",
        "row_major_3d_outer_slab_oob",
    }:
        shape = (2, 128, 128)
        layout = make_row_major(shape)
    elif copy_case == "zz_3d_major_dim_split":
        shape = (2, 96, 96)
        layout = make_zz_layout(shape, axes=[1, 2], block_shape=(32, 32))
    elif copy_case.startswith("dynamic_outer_"):
        shape = (T.dynamic("m"), 96)
        layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    elif copy_case == "zz_block_non_major_dim_multi_block":
        shape = (128, 128)
        layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    elif copy_case.startswith("zz_") or copy_case.startswith("zn_"):
        shape = (96, 96)
        if copy_case.startswith("zn_"):
            layout = make_zn_layout(shape, axes=[0, 1], block_shape=(32, 32))
        else:
            layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))
    else:
        shape = (128, 128)
        layout = make_row_major(shape)

    if copy_case == "row_major_aligned_grid_tile":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:64, 0:64], A_shared[64:128, 64:128])

    elif copy_case == "row_major_min_not_region_extent_aligned":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[32:96, 0:64], A_shared[0:64, 0:64])

    elif copy_case == "row_major_extent_not_buffer_shape_factor":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:96, 0:96], A_shared[0:96, 0:96])

    elif copy_case == "row_major_1d_tile_view":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0, 64:128], A_shared[1, 64:128])

    elif copy_case == "row_major_dynamic_min_bounds_unknown":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as cid:
                bx = cid % 4
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[bx * 32 : (bx + 1) * 32, 0:64], A_shared[0:32, 0:64])

    elif copy_case == "row_major_dynamic_min_alignment_unknown":
        shape = (512, 256)
        shared_shape = (32, 32)
        layout = make_row_major(shape)
        shared_layout = make_row_major(shared_shape)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as cid:
                bx = cid % 8
                A_shared = T.alloc_shared(shared_shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: shared_layout})
                ko = (bx + bx) % 8
                m_offset = bx * 32 + T.min(bx, 1)
                T.copy(A[m_offset, ko * 32], A_shared)

    elif copy_case == "row_major_dynamic_select_alignment_unknown":
        shape = (512, 512)
        shared_shape = (32, 32)
        layout = make_row_major(shape)
        shared_layout = make_row_major(shared_shape)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as cid:
                bx = cid % 4
                by = cid // 4
                A_shared = T.alloc_shared(shared_shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: shared_layout})
                sum_idx = bx + by
                diff_idx = bx - by
                scaled_idx = sum_idx * 3 + 1
                div_idx = scaled_idx // 2
                mod_idx = div_idx % 7
                lt_cmp = bx < by + 2
                le_cmp = by <= bx + 1
                ne_cmp = bx != by
                eq_cmp = bx == by
                not_lt_le = T.Not(lt_cmp and le_cmp)
                row_cond = (lt_cmp and le_cmp) or (ne_cmp and not_lt_le)
                row_delta = T.Select(
                    row_cond,
                    T.Select(eq_cmp, T.min(mod_idx, 7), T.min(mod_idx + 1, 7)),
                    T.max(diff_idx, 0),
                )
                src_row = by * 64 + 8 + row_delta
                T.copy(A[src_row, bx * 64], A_shared)

    elif copy_case == "row_major_3d_outer_slab_valid":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:2, 0:64, 0:64], A_shared[0:2, 64:128, 64:128])

    elif copy_case == "row_major_3d_outer_slab_unclipped":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[1:3, 0:64, 0:64], A_shared[0:2, 0:64, 0:64])

    elif copy_case == "row_major_3d_outer_slab_oob":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.evaluate(
                    tvm.tir.call_intrin(
                        "handle",
                        tvm.tir.op.Op.get("tl.tileop.copy"),
                        _tile_region(A[1, 0, 0], "r", 2, 64, 64),
                        _tile_region(A_shared[0, 0, 0], "w", 2, 64, 64),
                    )
                )

    elif copy_case == "row_major_zero_extent":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.evaluate(
                    tvm.tir.call_intrin(
                        "handle",
                        tvm.tir.op.Op.get("tl.tileop.copy"),
                        _tile_region(A[0, 0], "r", 0, 64),
                        _tile_region(A_shared[0, 0], "w", 0, 64),
                    )
                )

    elif copy_case == "row_major_negative_extent":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.evaluate(
                    tvm.tir.call_intrin(
                        "handle",
                        tvm.tir.op.Op.get("tl.tileop.copy"),
                        _tile_region(A[0, 0], "r", -16, 64),
                        _tile_region(A_shared[0, 0], "w", -16, 64),
                    )
                )

    elif copy_case == "zz_3d_major_dim_split":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:2, 0:16, 0:32], A_shared[0:2, 16:32, 32:64])

    elif copy_case == "zz_block_equal":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:32, 0:32], A_shared[32:64, 32:64])

    elif copy_case == "zz_block_non_major_dim_split":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:32, 0:16], A_shared[32:64, 16:32])

    elif copy_case == "zz_block_non_major_dim_multi_block":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:32, 0:64], A_shared[32:64, 0:64])

    elif copy_case == "zn_block_non_major_dim_split":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:16, 0:96], A_shared[16:32, 0:96])

    elif copy_case == "zz_block_both_dims_split":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:16, 0:16], A_shared[16:32, 16:32])

    elif copy_case == "zz_whole_dim":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:96, 0:96], A_shared[0:96, 0:96])

    elif copy_case == "zz_block_major_dim_split":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:16, 0:32], A_shared[16:32, 32:64])

    elif copy_case == "zz_block_offset_inside_tile":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[16:48, 0:32], A_shared[0:32, 32:64])

    elif copy_case == "zz_extent_not_coalesced_compatible":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:48, 0:48], A_shared[0:48, 0:48])

    elif copy_case == "dynamic_extent_warns_and_passes":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0 : shape[0], 0:32], A_shared[0 : shape[0], 32:64])

    elif copy_case == "dynamic_outer_static_extent_equal_block":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:32, 0:32], A_shared[32:64, 32:64])

    elif copy_case == "dynamic_outer_static_extent_split_block":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:16, 0:32], A_shared[16:32, 32:64])

    elif copy_case == "dynamic_outer_static_extent_incompatible":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:24, 0:32], A_shared[24:48, 32:64])

    elif copy_case == "dynamic_row_major_inner_mode_warns_and_passes":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A: layout, A_shared: layout})
                T.copy(A[0:32, 0:32], A_shared[0:32, 32:64])

    else:
        raise ValueError(f"unknown copy case: {copy_case}")

    return tvm.IRModule({"main": kernel})


def _run_validate_tile_view_regions(copy_case):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = _make_copy_kernel(copy_case)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        return tilelang.transform.ValidateTileViewRegions()(mod)


def _make_inferred_layout_copy_kernel(copy_case):
    shape = (96, 96) if copy_case.startswith("zz_") else (128, 128)

    @T.prim_func
    def kernel(A: T.Tensor(shape, DTYPE)):
        with T.Kernel():
            A_shared = T.alloc_shared(shape, DTYPE)

            if copy_case == "row_major_aligned_grid_tile":
                T.copy(A[0:64, 0:64], A_shared[64:128, 64:128])
            elif copy_case == "row_major_min_not_region_extent_aligned":
                T.copy(A[32:96, 0:64], A_shared[0:64, 0:64])
            elif copy_case == "zz_extent_not_coalesced_compatible":
                T.copy(A[0:48, 0:48], A_shared[0:48, 0:48])
            elif copy_case == "zz_block_equal":
                T.copy(A[0:32, 0:32], A_shared[32:64, 32:64])
            else:
                raise ValueError(f"unknown copy case: {copy_case}")

    return tvm.IRModule({"main": kernel})


def _run_kernel_pipeline_to_validate_tile_view_regions(copy_case):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = _make_inferred_layout_copy_kernel(copy_case)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.InferSramScope()(mod)
        mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
        mod = tilelang.transform.SunmmioLayoutInference()(mod)
        return tilelang.transform.ValidateTileViewRegions()(mod)


def _make_comm_kernel(comm_case):
    shape = (96, 96) if comm_case.endswith("_illegal_non_major_split") else (128, 128)
    layout = make_zz_layout(shape, axes=[0, 1], block_shape=(32, 32))

    if comm_case == "broadcast_valid":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                B_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A_shared: layout, B_shared: layout})
                T.comm.broadcast(
                    A_shared[0:32, 0:32],
                    B_shared[32:64, 32:64],
                    (0, 0),
                    direction="h",
                )

    elif comm_case == "broadcast_illegal_non_major_split":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                B_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A_shared: layout, B_shared: layout})
                T.comm.broadcast(
                    A_shared[0:32, 0:16],
                    B_shared[32:64, 16:32],
                    (0, 0),
                    direction="h",
                )

    elif comm_case == "broadcast_scalar_src_oob":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                B_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A_shared: layout, B_shared: layout})
                T.comm.broadcast(
                    A_shared[999, 0],
                    B_shared[0:32, 0:32],
                    (0, 0),
                    direction="h",
                )

    elif comm_case == "put_valid":

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel():
                A_shared = T.alloc_shared(shape, DTYPE)
                B_shared = T.alloc_shared(shape, DTYPE)
                T.annotate_layout({A_shared: layout, B_shared: layout})
                T.comm.put(
                    A_shared[0:32, 0:32],
                    B_shared[32:64, 32:64],
                    (0, 0),
                    (0, 1),
                )

    elif comm_case == "allgather_legacy_valid":
        recv_shape = (16, 32, 32)
        recv_layout = make_row_major(recv_shape)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as _bx:
                A_shared = T.alloc_shared((32, 32), DTYPE)
                R_shared = T.alloc_shared(recv_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major((32, 32)),
                        R_shared: recv_layout,
                    }
                )
                T.comm.all_gather(A_shared, R_shared, direction="all")

    elif comm_case == "allgather_axis_last_valid":
        send_shape = (32, 32)
        recv_shape = (32, 512)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as _bx:
                A_shared = T.alloc_shared(send_shape, DTYPE)
                R_shared = T.alloc_shared(recv_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        R_shared: make_row_major(recv_shape),
                    }
                )
                T.comm.all_gather(A_shared, R_shared, direction="all", axis=-1)

    elif comm_case == "allgather_axis0_valid":
        send_shape = (32, 32)
        recv_shape = (512, 32)
        recv_layout = make_zz_layout(recv_shape, axes=[0, 1], block_shape=(32, 32))

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as _bx:
                A_shared = T.alloc_shared(send_shape, DTYPE)
                R_shared = T.alloc_shared(recv_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        R_shared: recv_layout,
                    }
                )
                T.comm.all_gather(A_shared, R_shared, direction="all", axis=0)

    elif comm_case == "allgather_axis0_slot_extent_mismatch":
        send_shape = (32, 32)
        recv_shape = (544, 32)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as bx:
                A_shared = T.alloc_shared(send_shape, DTYPE)
                R_shared = T.alloc_shared(recv_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        R_shared: make_row_major(recv_shape),
                    }
                )
                T.evaluate(
                    tvm.tir.call_intrin(
                        "handle",
                        tvm.tir.op.Op.get("tl.tileop.comm_allgather"),
                        A_shared[0:32, 0:32],
                        R_shared[0:544, 0:32],
                        2,
                        -1,
                        0,
                        bx,
                    )
                )

    elif comm_case == "allgather_axis0_nonzero_recv_min_oob":
        send_shape = (32, 32)
        recv_shape = (512, 32)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as _bx:
                A_shared = T.alloc_shared(send_shape, DTYPE)
                R_shared = T.alloc_shared(recv_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        R_shared: make_row_major(recv_shape),
                    }
                )
                T.comm.all_gather(
                    A_shared,
                    R_shared[32:544, 0:32],
                    direction="all",
                    axis=0,
                )

    elif comm_case == "allgather_high_rank_valid":
        send_shape = (4, 4, 32, 32)
        recv_shape = (4, *send_shape)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as _bx:
                A_shared = T.alloc_shared(send_shape, DTYPE)
                R_shared = T.alloc_shared(recv_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        R_shared: make_row_major(recv_shape),
                    }
                )
                T.comm.all_gather(A_shared, R_shared, direction="horizontal")

    elif comm_case == "allreduce_low_level_valid":
        send_shape = (32, 32)
        out_shape = (32, 32)
        gather_shape = (4, 32, 32)

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as bx:
                A_shared = T.alloc_shared(send_shape, DTYPE)
                Out_shared = T.alloc_shared(out_shape, DTYPE)
                Row_gather = T.alloc_shared(gather_shape, DTYPE)
                Col_gather = T.alloc_shared(gather_shape, DTYPE)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        Out_shared: make_row_major(out_shape),
                        Row_gather: make_row_major(gather_shape),
                        Col_gather: make_row_major(gather_shape),
                    }
                )
                T.evaluate(
                    tvm.tir.call_intrin(
                        "handle",
                        tvm.tir.op.Op.get("tl.tileop.comm_allreduce"),
                        A_shared[0:32, 0:32],
                        Out_shared[0:32, 0:32],
                        Row_gather[0:4, 0:32, 0:32],
                        Col_gather[0:4, 0:32, 0:32],
                        "sum",
                        2,
                        T.bool(True),
                        bx,
                    )
                )

    elif comm_case in {
        "allreduce_low_level_mixed_dtype_valid",
        "allreduce_low_level_row_allgather_dtype_mismatch",
        "allreduce_low_level_col_allgather_dtype_mismatch",
        "allreduce_low_level_dst_copy_dtype_mismatch",
    }:
        send_shape = (32, 32)
        out_shape = (32, 32)
        gather_shape = (4, 32, 32)
        mismatched_name = (
            None
            if comm_case == "allreduce_low_level_mixed_dtype_valid"
            else comm_case.removeprefix("allreduce_low_level_").removesuffix("_dtype_mismatch")
        )
        out_dtype = "float32"
        row_dtype = DTYPE if mismatched_name == "row_allgather" else out_dtype
        col_dtype = out_dtype if mismatched_name == "col_allgather" else DTYPE
        copy_dtype = DTYPE if mismatched_name == "dst_copy" else out_dtype

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as bx:
                A_shared = T.alloc_shared(send_shape, DTYPE)
                Out_shared = T.alloc_shared(out_shape, out_dtype)
                Row_gather = T.alloc_shared(gather_shape, row_dtype)
                Col_gather = T.alloc_shared(gather_shape, col_dtype)
                Dst_copy = T.alloc_shared(out_shape, copy_dtype)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        Out_shared: make_row_major(out_shape),
                        Row_gather: make_row_major(gather_shape),
                        Col_gather: make_row_major(gather_shape),
                        Dst_copy: make_row_major(out_shape),
                    }
                )
                if mismatched_name == "dst_copy":
                    T.evaluate(
                        tvm.tir.call_intrin(
                            "handle",
                            tvm.tir.op.Op.get("tl.tileop.comm_allreduce"),
                            A_shared[0:32, 0:32],
                            Out_shared[0:32, 0:32],
                            Row_gather[0:4, 0:32, 0:32],
                            Col_gather[0:4, 0:32, 0:32],
                            "sum",
                            2,
                            T.bool(False),
                            Dst_copy[0:32, 0:32],
                            bx,
                        )
                    )
                else:
                    T.evaluate(
                        tvm.tir.call_intrin(
                            "handle",
                            tvm.tir.op.Op.get("tl.tileop.comm_allreduce"),
                            A_shared[0:32, 0:32],
                            Out_shared[0:32, 0:32],
                            Row_gather[0:4, 0:32, 0:32],
                            Col_gather[0:4, 0:32, 0:32],
                            "sum",
                            2,
                            T.bool(True),
                            bx,
                        )
                    )

    elif comm_case in {
        "allreduce_low_level_dst_shape_mismatch",
        "allreduce_low_level_row_allgather_shape_mismatch",
        "allreduce_low_level_col_allgather_shape_mismatch",
        "allreduce_low_level_dst_copy_shape_mismatch",
        "allreduce_low_level_missing_dst_copy",
        "allreduce_low_level_unexpected_dst_copy",
        "allreduce_low_level_bitwise_float",
        "allreduce_low_level_bitwise_int16",
        "allreduce_low_level_float16_minmax",
        "allreduce_low_level_float64_minmax",
        "allreduce_low_level_integer_minmax",
    }:
        send_shape = (32, 32)
        out_shape = (64, 16) if comm_case == "allreduce_low_level_dst_shape_mismatch" else send_shape
        row_shape = (4, 16, 64) if comm_case == "allreduce_low_level_row_allgather_shape_mismatch" else (4, *send_shape)
        col_shape = (4, 16, 64) if comm_case == "allreduce_low_level_col_allgather_shape_mismatch" else (4, *send_shape)
        copy_shape = (64, 16) if comm_case == "allreduce_low_level_dst_copy_shape_mismatch" else send_shape
        if comm_case in {
            "allreduce_low_level_row_allgather_shape_mismatch",
            "allreduce_low_level_unexpected_dst_copy",
            "allreduce_low_level_float16_minmax",
            "allreduce_low_level_float64_minmax",
            "allreduce_low_level_integer_minmax",
        }:
            direction = 0
        elif comm_case == "allreduce_low_level_col_allgather_shape_mismatch":
            direction = 1
        else:
            direction = 2
        clear = comm_case not in {
            "allreduce_low_level_dst_copy_shape_mismatch",
            "allreduce_low_level_missing_dst_copy",
            "allreduce_low_level_unexpected_dst_copy",
        }
        if comm_case in {
            "allreduce_low_level_bitwise_float",
            "allreduce_low_level_bitwise_int16",
        }:
            reduce_type = "bitand"
        elif comm_case in {
            "allreduce_low_level_float16_minmax",
            "allreduce_low_level_float64_minmax",
            "allreduce_low_level_integer_minmax",
        }:
            reduce_type = "min"
        else:
            reduce_type = "sum"
        if comm_case == "allreduce_low_level_float16_minmax":
            reduce_dtype = "float16"
        elif comm_case == "allreduce_low_level_float64_minmax":
            reduce_dtype = "float64"
        elif comm_case == "allreduce_low_level_integer_minmax":
            reduce_dtype = "int32"
        elif comm_case == "allreduce_low_level_bitwise_int16":
            reduce_dtype = "int16"
        else:
            reduce_dtype = DTYPE
        include_dst_copy = comm_case in {
            "allreduce_low_level_dst_copy_shape_mismatch",
            "allreduce_low_level_unexpected_dst_copy",
        }

        @T.prim_func
        def kernel(A: T.Tensor(shape, DTYPE)):
            with T.Kernel() as bx:
                A_shared = T.alloc_shared(send_shape, reduce_dtype)
                Out_shared = T.alloc_shared(out_shape, reduce_dtype)
                Row_gather = T.alloc_shared(row_shape, reduce_dtype)
                Col_gather = T.alloc_shared(col_shape, reduce_dtype)
                Dst_copy = T.alloc_shared(copy_shape, reduce_dtype)
                T.annotate_layout(
                    {
                        A_shared: make_row_major(send_shape),
                        Out_shared: make_row_major(out_shape),
                        Row_gather: make_row_major(row_shape),
                        Col_gather: make_row_major(col_shape),
                        Dst_copy: make_row_major(copy_shape),
                    }
                )
                if include_dst_copy:
                    T.evaluate(
                        tvm.tir.call_intrin(
                            "handle",
                            tvm.tir.op.Op.get("tl.tileop.comm_allreduce"),
                            A_shared[0:32, 0:32],
                            Out_shared[0 : out_shape[0], 0 : out_shape[1]],
                            Row_gather[0 : row_shape[0], 0 : row_shape[1], 0 : row_shape[2]],
                            Col_gather[0 : col_shape[0], 0 : col_shape[1], 0 : col_shape[2]],
                            reduce_type,
                            direction,
                            T.bool(clear),
                            Dst_copy[0 : copy_shape[0], 0 : copy_shape[1]],
                            bx,
                        )
                    )
                else:
                    T.evaluate(
                        tvm.tir.call_intrin(
                            "handle",
                            tvm.tir.op.Op.get("tl.tileop.comm_allreduce"),
                            A_shared[0:32, 0:32],
                            Out_shared[0 : out_shape[0], 0 : out_shape[1]],
                            Row_gather[0 : row_shape[0], 0 : row_shape[1], 0 : row_shape[2]],
                            Col_gather[0 : col_shape[0], 0 : col_shape[1], 0 : col_shape[2]],
                            reduce_type,
                            direction,
                            T.bool(clear),
                            bx,
                        )
                    )

    else:
        raise ValueError(f"unknown comm case: {comm_case}")

    return tvm.IRModule({"main": kernel})


def _run_comm_validate_tile_view_regions(comm_case):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = _make_comm_kernel(comm_case)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        return tilelang.transform.ValidateTileViewRegions()(mod)


def _run_comm_lower_tile_op(comm_case):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = _make_comm_kernel(comm_case)
        mod = tvm.tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.ResolveSunmmioMeshSymbols()(mod)
        mod = tilelang.transform.LegalizeNegativeIndex()(mod)
        mod = tilelang.transform.InjectAssumes()(mod)
        mod = tilelang.transform.Simplify()(mod)
        mod = tilelang.transform.InferSramScope()(mod)
        mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
        mod = tilelang.transform.SunmmioLayoutInference()(mod)
        mod = tilelang.transform.LegalizeSunmmioGemm()(mod)
        return tilelang.transform.LowerTileOp()(mod)


def _count_calls(mod, op_name):
    count = 0
    op = tvm.tir.op.Op.get(op_name)

    def visitor(node):
        nonlocal count
        if isinstance(node, tvm.tir.Call) and node.op.same_as(op):
            count += 1

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visitor)
    return count


@pytest.mark.parametrize(
    "copy_case",
    [
        "row_major_aligned_grid_tile",
        "row_major_1d_tile_view",
        "row_major_dynamic_min_bounds_unknown",
        "row_major_dynamic_min_alignment_unknown",
        "row_major_dynamic_select_alignment_unknown",
        "row_major_3d_outer_slab_valid",
        "zz_3d_major_dim_split",
        "zz_block_equal",
        "zz_block_non_major_dim_multi_block",
        "zz_block_major_dim_split",
        "zz_whole_dim",
        "dynamic_extent_warns_and_passes",
        "dynamic_row_major_inner_mode_warns_and_passes",
        "dynamic_outer_static_extent_equal_block",
        "dynamic_outer_static_extent_split_block",
    ],
)
def test_sunmmio_validate_tile_view_regions_accepts_legal_regions(copy_case):
    _run_validate_tile_view_regions(copy_case)


@pytest.mark.parametrize(
    "copy_case, error_msg",
    [
        (
            "row_major_min_not_region_extent_aligned",
            "must align to region extent",
        ),
        (
            "row_major_extent_not_buffer_shape_factor",
            "must divide buffer shape",
        ),
        (
            "row_major_3d_outer_slab_oob",
            "must stay within buffer shape",
        ),
        (
            "row_major_3d_outer_slab_unclipped",
            "must stay within buffer shape",
        ),
        (
            "row_major_zero_extent",
            "must be positive",
        ),
        (
            "row_major_negative_extent",
            "must be positive",
        ),
        (
            "zz_block_both_dims_split",
            "must not split the non-major dimension",
        ),
        (
            "zz_block_non_major_dim_split",
            "must not split the non-major dimension",
        ),
        (
            "zn_block_non_major_dim_split",
            "must not split the non-major dimension",
        ),
        (
            "zz_block_offset_inside_tile",
            "must align to region extent",
        ),
        (
            "zz_extent_not_coalesced_compatible",
            "must be compatible with coalesced extent",
        ),
        (
            "dynamic_outer_static_extent_incompatible",
            "must be compatible with dynamic layout inner static mode shape",
        ),
    ],
)
def test_sunmmio_validate_tile_view_regions_rejects_illegal_regions(copy_case, error_msg):
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_validate_tile_view_regions(copy_case)


def test_sunmmio_validate_tile_view_regions_accepts_kernel_after_layout_inference():
    _run_kernel_pipeline_to_validate_tile_view_regions("zz_block_equal")


@pytest.mark.parametrize(
    "copy_case, error_msg",
    [
        (
            "row_major_min_not_region_extent_aligned",
            "must align to region extent",
        ),
        (
            "zz_extent_not_coalesced_compatible",
            "must be compatible with coalesced extent",
        ),
    ],
)
def test_sunmmio_validate_tile_view_regions_rejects_kernel_after_layout_inference(copy_case, error_msg):
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_kernel_pipeline_to_validate_tile_view_regions(copy_case)


@pytest.mark.parametrize(
    "comm_case",
    [
        "broadcast_valid",
        "put_valid",
        "allgather_legacy_valid",
        "allgather_axis0_valid",
        "allgather_axis_last_valid",
        "allreduce_low_level_valid",
        "allreduce_low_level_mixed_dtype_valid",
    ],
)
def test_sunmmio_validate_tile_view_regions_accepts_comm_regions(comm_case):
    _run_comm_validate_tile_view_regions(comm_case)


def test_sunmmio_lower_tile_op_accepts_valid_allreduce_contract():
    mod = _run_comm_lower_tile_op("allreduce_low_level_valid")

    assert _count_calls(mod, "tl.tileop.comm_allreduce") == 0
    assert _count_calls(mod, "tl.broadcast_") > 0


def test_sunmmio_high_rank_allgather_uses_bounded_static_expansion():
    mod = _run_comm_lower_tile_op("allgather_high_rank_valid")
    split_loops = []

    def visitor(node):
        if isinstance(node, tvm.tir.For) and node.loop_var.name.startswith("allgather_split_"):
            split_loops.append(node)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visitor)
    assert _count_calls(mod, "tl.broadcast_") == 16
    assert not split_loops


@pytest.mark.parametrize(
    "comm_case, error_msg",
    [
        (
            "broadcast_illegal_non_major_split",
            "must not split the non-major dimension",
        ),
        (
            "allgather_axis0_slot_extent_mismatch",
            "must equal send extent",
        ),
    ],
)
def test_sunmmio_validate_tile_view_regions_rejects_comm_regions(comm_case, error_msg):
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_comm_validate_tile_view_regions(comm_case)


@pytest.mark.parametrize(
    "buffer_name,expected_buffer",
    [
        ("row_allgather", "dst"),
        ("col_allgather", "src"),
        ("dst_copy", "dst"),
    ],
)
def test_sunmmio_validate_tile_view_regions_rejects_allreduce_dtype_mismatch(buffer_name, expected_buffer):
    comm_case = f"allreduce_low_level_{buffer_name}_dtype_mismatch"
    error_msg = rf"requires {buffer_name} dtype to match {expected_buffer} dtype"
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_comm_validate_tile_view_regions(comm_case)


@pytest.mark.parametrize(
    "buffer_name,expected_buffer",
    [
        ("row_allgather", "dst"),
        ("col_allgather", "src"),
        ("dst_copy", "dst"),
    ],
)
def test_sunmmio_lower_tile_op_rejects_allreduce_dtype_mismatch(buffer_name, expected_buffer):
    comm_case = f"allreduce_low_level_{buffer_name}_dtype_mismatch"
    error_msg = rf"requires {buffer_name} dtype to match {expected_buffer} dtype"
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_comm_lower_tile_op(comm_case)


@pytest.mark.parametrize(
    "comm_case,error_msg",
    [
        ("allreduce_low_level_dst_shape_mismatch", "requires dst extent"),
        ("allreduce_low_level_row_allgather_shape_mismatch", "requires row_allgather payload extent"),
        ("allreduce_low_level_col_allgather_shape_mismatch", "requires col_allgather payload extent"),
        ("allreduce_low_level_dst_copy_shape_mismatch", "requires dst_copy extent"),
        ("allreduce_low_level_missing_dst_copy", "dst_copy must be provided only"),
        ("allreduce_low_level_unexpected_dst_copy", "dst_copy must be provided only"),
        (
            "allreduce_low_level_bitwise_float",
            "Sunmmio bitwise all_reduce requires input and output dtypes to be both int32 or both uint32",
        ),
        (
            "allreduce_low_level_bitwise_int16",
            "Sunmmio bitwise all_reduce requires input and output dtypes to be both int32 or both uint32",
        ),
        (
            "allreduce_low_level_float16_minmax",
            "Sunmmio all_reduce min/max/absmax requires scalar bfloat16 or float32 input and output dtypes",
        ),
        (
            "allreduce_low_level_float64_minmax",
            "Sunmmio all_reduce min/max/absmax requires scalar bfloat16 or float32 input and output dtypes",
        ),
        (
            "allreduce_low_level_integer_minmax",
            "Sunmmio all_reduce min/max/absmax requires scalar bfloat16 or float32 input and output dtypes",
        ),
    ],
)
@pytest.mark.parametrize("runner", [_run_comm_validate_tile_view_regions, _run_comm_lower_tile_op])
def test_sunmmio_rejects_invalid_low_level_allreduce_contract(comm_case, error_msg, runner):
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        runner(comm_case)


def test_sunmmio_comm_frontend_rejects_broadcast_source_oob(_strict_region_validation):
    with pytest.raises(ValueError, match="region starts outside buffer shape"):
        _make_comm_kernel("broadcast_scalar_src_oob")


def test_sunmmio_comm_frontend_clips_then_rejects_allgather_recv_oob(_strict_region_validation):
    with (
        pytest.warns(UserWarning, match="explicit BufferRegion exceeds buffer shape and will be clipped"),
        pytest.raises(ValueError, match="src extent is larger than dst extent"),
    ):
        _make_comm_kernel("allgather_axis0_nonzero_recv_min_oob")


if __name__ == "__main__":
    tilelang.testing.main()
