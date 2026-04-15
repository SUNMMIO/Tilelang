from testing.python.transform._sunmmio_tile_loop_fusion_test_utils import *


def _get_block_by_name(stmt, name):
    found = None

    def visit(node):
        nonlocal found
        if isinstance(node, tvm.tir.Block) and node.name_hint == name:
            found = node

    tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    assert found is not None, f"Expected block `{name}` in rewritten TIR"
    return found


def _as_seq(stmt):
    if isinstance(stmt, tvm.tir.SeqStmt):
        return list(stmt.seq)
    return [stmt]


def _root_seq(mod):
    root = _get_block_by_name(mod["main"].body, "root")
    return _as_seq(root.body)


def _tilelang_root_seq(mod):
    tilelang_root = _get_block_by_name(mod["main"].body, "tilelang_root")
    return _as_seq(tilelang_root.body)


def _for_annotations(loop):
    return dict(loop.annotations) if loop.annotations else {}


def _is_scope_entry_loop(stmt, *, tile_size=None, extent=None):
    if not isinstance(stmt, tvm.tir.For):
        return False
    annotations = _for_annotations(stmt)
    if "tile.scope_entry" not in annotations:
        return False
    if tile_size is not None and list(annotations.get("tile.tile_size", [])) != list(tile_size):
        return False
    if extent is not None and int(stmt.extent) != extent:  # noqa: SIM103
        return False
    return True


def _find_scope_entry_loops(stmts, *, tile_size=None, extent=None):
    return [stmt for stmt in stmts if _is_scope_entry_loop(stmt, tile_size=tile_size, extent=extent)]


def _expect_single_match(items, predicate, description):
    matches = [item for item in items if predicate(item)]
    assert len(matches) == 1, f"Expected exactly one {description}, but found {len(matches)}"
    return matches[0]


def _expect_loop_body_seq(scope_loop):
    body = scope_loop.body
    if isinstance(body, tvm.tir.For) and _for_annotations(body).get("tile.execution_axis") == 1:
        return _as_seq(body.body)
    return _as_seq(body)


def _expect_reduce_block(stmt):
    assert isinstance(stmt, tvm.tir.Block)
    assert stmt.name_hint == "reduce_tile_op"
    return stmt


def _collect_buffer_accesses(stmt):
    reads = []
    writes = []

    def visit(node):
        if isinstance(node, tvm.tir.BufferLoad):
            reads.append(node.buffer.name)
        elif isinstance(node, tvm.tir.BufferStore):
            writes.append(node.buffer.name)

    tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    return reads, writes


def _semantic_leaf_tag(stmt):
    if isinstance(stmt, tvm.tir.LetStmt):
        return _semantic_leaf_tag(stmt.body)
    if isinstance(stmt, tvm.tir.AttrStmt):
        return _semantic_leaf_tag(stmt.body)
    if isinstance(stmt, tvm.tir.Block):
        assert stmt.name_hint == "reduce_tile_op"
        allocs = {buf.name for buf in stmt.alloc_buffers}
        if "scores_sum_acc" in allocs:
            return "reduce_scores_sum"
        if "scores_max_acc" in allocs:
            return "reduce_scores_max"
        if "row_sum_acc" in allocs:
            return "reduce_row_sum"
        raise AssertionError(f"Unknown reduction block alloc buffers: {sorted(allocs)}")

    reads, writes = _collect_buffer_accesses(stmt)
    write_set = set(writes)
    if write_set == {"Tmp_shared"}:
        return "Tmp_shared"
    if write_set == {"B_shared"}:
        return "B_shared"
    if write_set == {"scores_max_prev"}:
        return "scores_max_prev"
    if write_set == {"scores_max"}:
        return "scores_max"
    if write_set == {"scores_scale"}:
        return "scores_scale"
    if write_set == {"acc_s"}:
        return "acc_s"
    if write_set == {"acc_s_cast"}:
        return "acc_s_cast"
    if write_set == {"a_square"}:
        return "a_square"
    if write_set == {"acc_o"}:
        return "acc_o"
    if write_set == {"row_sum"}:
        return "row_sum_init"
    if write_set == {"row_scale"}:
        return "row_scale"
    if write_set == {"logsum"}:
        return "logsum"
    if write_set == {"a_out"}:
        return "a_out"
    raise AssertionError(f"Unknown semantic leaf writes={sorted(write_set)} reads={sorted(set(reads))}")


def _semantic_tags(stmts):
    return [_semantic_leaf_tag(stmt) for stmt in stmts]


def test_sunmmio_tile_loop_fusion_pass_is_noop_for_stage1():
    mod = IRModule.from_expr(single_tile_scope_kernel().with_attr("global_symbol", "main"))
    mod = apply_tiles_lowering(mod)

    before = mod.script()
    after = tl.transform.SunmmioTileLoopFusion()(mod)

    assert after.script() == before


def test_sunmmio_tile_loop_fusion_rewrites_consecutive_tile_scopes():
    mod = IRModule.from_expr(two_consecutive_tile_scopes_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tile_loop_fusion(mod)

    summary = get_debug_summary(mod)
    stmts = _tilelang_root_seq(mod)
    fused_loop = _expect_single_match(
        _find_scope_entry_loops(stmts, tile_size=[8, 32], extent=4),
        lambda loop: _semantic_tags(_expect_loop_body_seq(loop)) == ["Tmp_shared", "B_shared"],
        "fused [8, 32] tile shell with Tmp_shared then B_shared semantics",
    )

    assert summary["region_count"] == 1
    assert summary["window_count"] == 1
    assert summary["window_lengths"] == [1]
    assert _semantic_tags(_expect_loop_body_seq(fused_loop)) == ["Tmp_shared", "B_shared"]


def test_sunmmio_tile_loop_fusion_rewrites_flash_attention_window():
    mod = IRModule.from_expr(flash_attention_online_softmax_tiled_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tile_loop_fusion(mod)

    summary = get_debug_summary(mod)
    stmts = _tilelang_root_seq(mod)
    fused_row = _expect_single_match(
        _find_scope_entry_loops(stmts, tile_size=[32], extent=1),
        lambda loop: _semantic_tags(_expect_loop_body_seq(loop)) == ["scores_max", "scores_scale"],
        "fused flash row shell realizing scores_max then scores_scale",
    )
    fused_tile = _expect_single_match(
        _find_scope_entry_loops(stmts, tile_size=[4, 32], extent=8),
        lambda loop: _semantic_tags(_expect_loop_body_seq(loop)) == ["acc_s", "acc_s_cast", "reduce_scores_sum"],
        "fused flash tile shell realizing acc_s, acc_s_cast, then reduce_scores_sum",
    )

    assert summary["region_count"] == 7
    assert summary["window_count"] == 1
    assert summary["window_lengths"] == [7]
    assert _semantic_tags(_expect_loop_body_seq(fused_row)) == ["scores_max", "scores_scale"]
    assert _semantic_tags(_expect_loop_body_seq(fused_tile)) == ["acc_s", "acc_s_cast", "reduce_scores_sum"]


def test_sunmmio_tile_loop_fusion_rewrite_preserves_flash_reduction_local_scratch():
    mod = IRModule.from_expr(flash_attention_online_softmax_tiled_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tile_loop_fusion(mod)

    stmts = _tilelang_root_seq(mod)

    top_level_reduce_block = _expect_single_match(
        stmts,
        lambda stmt: isinstance(stmt, tvm.tir.Block) and stmt.name_hint == "reduce_tile_op",
        "top-level flash reduce_max block",
    )

    fused_tile = _expect_single_match(
        _find_scope_entry_loops(stmts, tile_size=[4, 32], extent=8),
        lambda loop: _semantic_tags(_expect_loop_body_seq(loop)) == ["acc_s", "acc_s_cast", "reduce_scores_sum"],
        "flash fused tile shell with local reduce_scores_sum block",
    )
    fused_reduce = _expect_reduce_block(_expect_loop_body_seq(fused_tile)[2])
    fused_reduce_body = _as_seq(fused_reduce.body)

    assert isinstance(fused_reduce_body[0], tvm.tir.IfThenElse)
    assert isinstance(fused_reduce_body[1], tvm.tir.For)
    assert isinstance(fused_reduce_body[2], tvm.tir.IfThenElse)

    top_level_allocs = [buf.name for buf in top_level_reduce_block.alloc_buffers]
    fused_allocs = [buf.name for buf in fused_reduce.alloc_buffers]

    assert top_level_allocs.count("scores_max_acc") == 1
    assert top_level_allocs.count("scores_max_res") == 1
    assert fused_allocs.count("scores_sum_acc") == 1


def test_sunmmio_tile_loop_fusion_rewrite_hoists_common_attr_wrapper():
    mod = IRModule.from_expr(attr_wrapped_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    summary = get_debug_summary(mod)
    root_stmts = _root_seq(mod)

    assert summary["region_count"] == 1
    assert summary["window_count"] == 1
    assert summary["window_lengths"] == [1]

    attr_stmt = _expect_single_match(
        root_stmts,
        lambda stmt: isinstance(stmt, tvm.tir.AttrStmt),
        "hoisted AttrStmt wrapper",
    )
    assert int(attr_stmt.value) == 1
    fused_loop = _expect_single_match(
        _as_seq(attr_stmt.body),
        lambda stmt: _is_scope_entry_loop(stmt, tile_size=[8, 32], extent=4),
        "fused child under hoisted AttrStmt",
    )
    assert _semantic_tags(_expect_loop_body_seq(fused_loop)) == ["Tmp_shared", "B_shared"]


def test_sunmmio_tile_loop_fusion_rewrite_preserves_local_let_wrappers():
    mod = IRModule.from_expr(let_wrapped_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    summary = get_debug_summary(mod)
    root_stmts = _root_seq(mod)

    assert summary["region_count"] == 1
    assert summary["window_count"] == 1
    assert summary["window_lengths"] == [1]

    let_stmt = _expect_single_match(
        root_stmts,
        lambda stmt: isinstance(stmt, tvm.tir.LetStmt),
        "outer local LetStmt wrapper",
    )
    assert let_stmt.var.name == "x0"
    assert int(let_stmt.value) == 7
    fused_loop = let_stmt.body
    assert _is_scope_entry_loop(fused_loop, tile_size=[8, 32], extent=4)
    fused_body = _expect_loop_body_seq(fused_loop)
    assert _semantic_leaf_tag(fused_body[0]) == "Tmp_shared"
    assert isinstance(fused_body[1], tvm.tir.LetStmt)
    assert fused_body[1].var.name == "x1"
    assert int(fused_body[1].value) == 11
    assert _semantic_leaf_tag(fused_body[1]) == "B_shared"


def test_sunmmio_tile_loop_fusion_rewrite_preserves_local_let_in_mixed_cluster():
    mod = IRModule.from_expr(mixed_plain_and_let_wrapped_two_region_lowered_kernel().with_attr("global_symbol", "main"))
    mod = tl.transform.SunmmioTileLoopFusion()(mod)

    summary = get_debug_summary(mod)
    root_stmts = _root_seq(mod)

    assert summary["region_count"] == 1
    assert summary["window_count"] == 1
    assert summary["window_lengths"] == [1]

    fused_loop = _expect_single_match(
        root_stmts,
        lambda stmt: _is_scope_entry_loop(stmt, tile_size=[8, 32], extent=4),
        "mixed fused shell",
    )
    fused_body = _expect_loop_body_seq(fused_loop)
    assert _semantic_leaf_tag(fused_body[0]) == "Tmp_shared"
    assert isinstance(fused_body[1], tvm.tir.LetStmt)
    assert fused_body[1].var.name == "x1"
    assert int(fused_body[1].value) == 7
    assert _semantic_leaf_tag(fused_body[1]) == "B_shared"


def test_sunmmio_tile_loop_fusion_rewrites_rmsnorm_window_structurally():
    mod = IRModule.from_expr(rms_norm_tiled_kernel().with_attr("global_symbol", "main"))
    mod = apply_sunmmio_tile_loop_fusion(mod)

    summary = get_debug_summary(mod)
    stmts = _tilelang_root_seq(mod)

    assert summary["region_count"] == 4
    assert summary["window_count"] == 1
    assert summary["window_lengths"] == [4]

    fused_tile = _expect_single_match(
        _find_scope_entry_loops(stmts, tile_size=[4, 32], extent=8),
        lambda loop: _semantic_tags(_expect_loop_body_seq(loop)) == ["a_square", "reduce_row_sum"],
        "fused RMSNorm tile shell realizing a_square then reduce_row_sum",
    )
    fused_tile_body = _expect_loop_body_seq(fused_tile)
    assert _semantic_tags(fused_tile_body) == ["a_square", "reduce_row_sum"]
    reduce_block = _expect_reduce_block(fused_tile_body[1])
    reduce_block_body = _as_seq(reduce_block.body)
    assert len(reduce_block_body) == 3
    assert isinstance(reduce_block_body[0], tvm.tir.IfThenElse)
    assert isinstance(reduce_block_body[1], tvm.tir.For)
    assert isinstance(reduce_block_body[2], tvm.tir.IfThenElse)
