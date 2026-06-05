import os

import tilelang.testing
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target

from compile_pipeline import target
from sunmmio_codegen_validation_utils import (
    assert_source_contains,
    validate_suvm_mlir_with_npuir_opt,
    write_sunmmio_codegen_logs,
)


os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"


def _to_device_kernel_func(func):
    return func.with_attr("global_symbol", "main").with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))


def _primfunc_from_stmt(stmt, params=None):
    if params is None:
        params = []
    return _to_device_kernel_func(tvm.tir.PrimFunc(params, stmt))


def _build_sunmmio_source_from_stmt(stmt, params=None):
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _primfunc_from_stmt(stmt, params=params)})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return mod, builder(mod, target, "suvm").inspect_source()


def _validate_stmt_codegen(stmt, tmp_path, *, mlir_filename, expected_tokens=(), params=None):
    tir_mod, src = _build_sunmmio_source_from_stmt(stmt, params=params)
    assert_source_contains(src, ("module", "suvm.device_arch", "func.func @", *expected_tokens))
    write_sunmmio_codegen_logs(
        case_name=mlir_filename,
        tir_mod=tir_mod,
        mlir_src=src,
    )
    validate_suvm_mlir_with_npuir_opt(src, tmp_path, mlir_filename=mlir_filename)
    return src


def _shared_buffers(dtype="float16", shape=(32, 32)):
    elem_ty = tvm.ir.PrimType(dtype)
    src_data = tvm.tir.Var("src_data", tvm.ir.PointerType(elem_ty, "shared.rsram"))
    dst_data = tvm.tir.Var("dst_data", tvm.ir.PointerType(elem_ty, "shared.asram"))
    src_buf = tvm.tir.decl_buffer(shape, dtype, name="Src", data=src_data, scope="shared.rsram")
    dst_buf = tvm.tir.decl_buffer(shape, dtype, name="Dst", data=dst_data, scope="shared.asram")
    return src_data, dst_data, src_buf, dst_buf


def _region(buf, access, row=None, col=None, extents=(32, 32)):
    if row is None:
        row = tvm.tir.IntImm("int32", 0)
    if col is None:
        col = tvm.tir.IntImm("int32", 0)
    args = [
        tvm.tir.BufferLoad(buf, [row, col]),
        tvm.tir.IntImm("int32", access),
        *[tvm.tir.IntImm("int32", extent) for extent in extents],
    ]
    return tvm.tir.call_intrin("handle", tvm.ir.Op.get("tl.tileop.region"), *args)


def _sync_token(token_id):
    return tvm.tir.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.sync_token_id"),
        tvm.tir.IntImm("int32", token_id),
    )


def _sync_null_token(token_id):
    return tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.sync_null_token"),
        [tvm.tir.IntImm("int32", token_id)],
    )


def _wait_token(token_id):
    return tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.wait_token"),
        [tvm.tir.IntImm("int32", token_id)],
    )


def _broadcast(src_buf, dst_buf, *, direction=0, mask=None, src_core=None, token_id=None):
    if mask is None:
        mask = tvm.tir.IntImm("int64", 15)
    args = [
        _region(src_buf, 1),
        _region(dst_buf, 2),
        tvm.tir.IntImm("int32", direction),
        mask,
        tvm.tir.IntImm("int32", 0),
    ]
    if src_core is not None:
        args.append(src_core)
    if token_id is not None:
        args.append(_sync_token(token_id))
    return tvm.tir.Call("handle", tvm.ir.Op.get("tl.broadcast_"), args)


def _with_decl_buffers(stmt, buffers):
    for buf in reversed(buffers):
        stmt = tvm.tir.DeclBuffer(buf, stmt)
    return stmt


@target("Sunmmio")
def _broadcast_stmt(*, direction=0, mask=None, src_core=None, token_id=None, wait=False):
    src_data, dst_data, src_buf, dst_buf = _shared_buffers()
    stmts = [tvm.tir.Evaluate(_broadcast(src_buf, dst_buf, direction=direction, mask=mask, src_core=src_core, token_id=token_id))]
    if wait:
        stmts.append(tvm.tir.Evaluate(_wait_token(token_id)))
    body = tvm.tir.SeqStmt(stmts) if len(stmts) > 1 else stmts[0]
    return _with_decl_buffers(body, [src_buf, dst_buf]), [src_data, dst_data]


def test_broadcast_static_mask_token_codegen_validates_with_npuir_opt(tmp_path):
    stmt, params = _broadcast_stmt(token_id=0, wait=True)
    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        params=params,
        mlir_filename="broadcast_static_mask_token_suvm.mlir",
        expected_tokens=("suvm.get_partitioned_tile_view", "suvm.mcast_tok", "suvm.wait_token"),
    )

    assert "sunmmio.fake" not in src
    assert src.count("suvm.mcast_tok") == 1


def test_broadcast_dynamic_mask_codegen_validates_with_npuir_opt(tmp_path):
    src_data, dst_data, src_buf, dst_buf = _shared_buffers()
    bx = tvm.tir.Var("bx", "int32")
    bx_i64 = tvm.tir.Cast("int64", bx)
    one = tvm.tir.IntImm("int64", 1)
    mask = tvm.tir.bitwise_or(
        tvm.tir.shift_left(one, bx_i64),
        tvm.tir.shift_left(one, bx_i64 + tvm.tir.IntImm("int64", 1)),
    )
    body = tvm.tir.Evaluate(_broadcast(src_buf, dst_buf, mask=mask, token_id=1))
    stmt = tvm.tir.For(bx, 0, 3, tvm.tir.ForKind.SERIAL, body)
    stmt = _with_decl_buffers(stmt, [src_buf, dst_buf])

    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        params=[src_data, dst_data],
        mlir_filename="broadcast_dynamic_mask_suvm.mlir",
        expected_tokens=("scf.for", "arith.shli", "arith.ori", "suvm.mcast_tok"),
    )

    assert "sunmmio.fake" not in src


def test_broadcast_with_src_core_guards_mcast_codegen_validates_with_npuir_opt(tmp_path):
    stmt, params = _broadcast_stmt(src_core=tvm.tir.IntImm("int32", 0), token_id=2, wait=True)
    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        params=params,
        mlir_filename="broadcast_src_core_guard_suvm.mlir",
        expected_tokens=(
            "suvm.get_core_id",
            "arith.cmpi eq",
            "scf.if",
            "suvm.mcast_tok",
            "suvm.null_token",
            "scf.yield",
            "suvm.wait_token",
        ),
    )

    assert "sunmmio.fake" not in src


def test_static_barrier_reuses_single_init_codegen_validates_with_npuir_opt(tmp_path):
    mask = tvm.tir.IntImm("int64", 15)
    barrier_init = tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_init"), [mask])
    barrier_wait = tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_arrive_and_wait"), [mask])
    stmt = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(barrier_init),
            tvm.tir.Evaluate(barrier_wait),
            tvm.tir.Evaluate(barrier_wait),
        ]
    )

    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        mlir_filename="static_barrier_reuse_suvm.mlir",
        expected_tokens=("suvm.barrier.init", "suvm.barrier.arrive_and_wait"),
    )

    assert "suvm.barrier.init mask = 15 : !suvm.barrier" in src
    assert src.count("suvm.barrier.init") == 1
    assert src.count("suvm.barrier.arrive_and_wait") == 2


def test_dynamic_barrier_candidates_codegen_validates_with_npuir_opt(tmp_path):
    bx = tvm.tir.Var("bx", "int32")
    bx_i64 = tvm.tir.Cast("int64", bx)
    mask = tvm.tir.shift_left(
        tvm.tir.IntImm("int64", 15),
        bx_i64 * tvm.tir.IntImm("int64", 4),
    )
    candidates = [15, 240, 3840, 61440]
    barrier_init = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.barrier_init"),
        [tvm.tir.IntImm("int64", -1)] + [tvm.tir.IntImm("int64", candidate) for candidate in candidates],
    )
    barrier_wait = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.barrier_arrive_and_wait"),
        [mask] + [tvm.tir.IntImm("int64", candidate) for candidate in candidates],
    )
    stmt = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(barrier_init),
            tvm.tir.For(bx, 0, 4, tvm.tir.ForKind.SERIAL, tvm.tir.Evaluate(barrier_wait)),
        ]
    )

    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        mlir_filename="dynamic_barrier_candidates_suvm.mlir",
        expected_tokens=("scf.for", "scf.if", "arith.shli", "arith.cmpi eq", "cf.assert"),
    )

    for candidate in candidates:
        assert f"suvm.barrier.init mask = {candidate} : !suvm.barrier" in src
    assert src.count("suvm.barrier.init") == len(candidates)
    assert src.count("suvm.barrier.arrive_and_wait") == len(candidates)


def test_for_loop_carries_token_codegen_validates_with_npuir_opt(tmp_path):
    i = tvm.tir.Var("i", "int32")
    loop_body = tvm.tir.Evaluate(_sync_null_token(3))
    stmt = tvm.tir.SeqStmt(
        [
            tvm.tir.For(i, 0, 2, tvm.tir.ForKind.SERIAL, loop_body),
            tvm.tir.Evaluate(_wait_token(3)),
        ]
    )

    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        mlir_filename="for_loop_token_live_out_suvm.mlir",
        expected_tokens=("scf.for", "!suvm.token", "suvm.null_token", "scf.yield", "suvm.wait_token"),
    )

    assert src.count("suvm.wait_token") == 1


def test_if_token_merge_codegen_validates_with_npuir_opt(tmp_path):
    i = tvm.tir.Var("i", "int32")
    cond = tvm.tir.LT(i, tvm.tir.IntImm("int32", 1))
    then_case = tvm.tir.Evaluate(_sync_null_token(4))
    else_case = tvm.tir.Evaluate(_sync_null_token(4))
    stmt = tvm.tir.SeqStmt(
        [
            tvm.tir.IfThenElse(cond, then_case, else_case),
            tvm.tir.Evaluate(_wait_token(4)),
        ]
    )

    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        mlir_filename="if_token_merge_suvm.mlir",
        expected_tokens=("scf.if", "!suvm.token", "suvm.null_token", "scf.yield", "suvm.wait_token"),
    )

    assert src.count("scf.yield") >= 2


def test_while_token_carried_codegen_validates_with_npuir_opt(tmp_path):
    cond = tvm.tir.LT(tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1))
    stmt = tvm.tir.SeqStmt(
        [
            tvm.tir.While(cond, tvm.tir.Evaluate(_sync_null_token(5))),
            tvm.tir.Evaluate(_wait_token(5)),
        ]
    )

    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        mlir_filename="while_token_live_out_suvm.mlir",
        expected_tokens=("scf.while", "scf.condition", "!suvm.token", "suvm.null_token", "scf.yield", "suvm.wait_token"),
    )

    assert "sunmmio.fake" not in src


def test_nested_for_if_while_barrier_codegen_validates_with_npuir_opt(tmp_path):
    i = tvm.tir.Var("i", "int32")
    cond = tvm.tir.LT(i, tvm.tir.IntImm("int32", 1))
    while_cond = tvm.tir.LT(tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1))
    barrier_init = tvm.tir.Evaluate(tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_init"), [tvm.tir.IntImm("int64", 15)]))
    barrier_wait = tvm.tir.Evaluate(tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_arrive_and_wait"), [tvm.tir.IntImm("int64", 15)]))
    loop_body = tvm.tir.IfThenElse(
        cond,
        tvm.tir.While(while_cond, barrier_wait),
        tvm.tir.Evaluate(_sync_null_token(6)),
    )
    stmt = tvm.tir.SeqStmt(
        [
            barrier_init,
            tvm.tir.For(i, 0, 1, tvm.tir.ForKind.SERIAL, loop_body),
        ]
    )

    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        mlir_filename="nested_for_if_while_barrier_suvm.mlir",
        expected_tokens=("scf.for", "scf.if", "scf.while", "suvm.barrier.arrive_and_wait"),
    )

    assert "suvm.barrier.init mask = 15 : !suvm.barrier" in src


if __name__ == "__main__":
    tilelang.testing.main()
