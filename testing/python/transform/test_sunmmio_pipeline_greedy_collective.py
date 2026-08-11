import json

import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import should_force_let_inline
from tilelang.layout import make_zz_layout
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from testing.python.sunmmio.common.compile_pipeline import target
from tvm import tir


@target("Sunmmio")
def mesh_ffn_new(
    seq=128,
    hidden=512,
    inner_dim=512,
    block_seq=32,
    block_hidden=32,
    block_inner=32,
    num_stages=2,
    dtype="bfloat16",
    accum_dtype="float",
):
    activation_policy = T.MeshShardingPolicy(y=0, x=1)
    weight_policy = T.MeshShardingPolicy(y=0, x=1)
    x_shape = (seq, hidden)
    up_weight_shape = (hidden, inner_dim)
    mid_shape = (seq, inner_dim)
    down_weight_shape = (inner_dim, hidden)

    @T.prim_func
    def main(
        X: T.MeshTensor(x_shape, activation_policy, dtype, layout=make_zz_layout(x_shape)),
        WUp: T.MeshTensor(up_weight_shape, weight_policy, dtype, layout=make_zz_layout(up_weight_shape)),
        WDown: T.MeshTensor(down_weight_shape, weight_policy, dtype, layout=make_zz_layout(down_weight_shape)),
        Mid: T.MeshTensor(mid_shape, activation_policy, dtype, layout=make_zz_layout(mid_shape)),
        Y: T.MeshTensor(x_shape, activation_policy, accum_dtype, layout=make_zz_layout(x_shape)),
    ):
        with T.Kernel(T.mesh_ncores()):
            lhs_local = T.alloc_shared((block_seq, block_hidden), dtype, scope="shared.rsram")
            up_local = T.alloc_shared((block_hidden, block_inner), dtype, scope="shared.rsram")
            lhs_shared = T.alloc_shared((block_seq, block_hidden * T.mesh_ncols()), dtype)
            up_shared = T.alloc_shared((block_hidden * T.mesh_nrows(), block_inner), dtype)
            mid_acc = T.alloc_shared((block_seq, block_inner), accum_dtype, scope="shared.rsram")
            mid_tile = T.alloc_shared((block_seq, block_inner), dtype, scope="shared.rsram")
            mid_local = T.alloc_shared((block_seq, block_inner), dtype, scope="shared.rsram")
            down_local = T.alloc_shared((block_inner, block_hidden), dtype, scope="shared.rsram")
            mid_shared = T.alloc_shared((block_seq, block_inner * T.mesh_ncols()), dtype)
            down_shared = T.alloc_shared((block_inner * T.mesh_nrows(), block_hidden), dtype)
            out_acc = T.alloc_shared((block_seq, block_hidden), accum_dtype, scope="shared.rsram")

            hidden_blocks = T.ceildiv(X.local_shape[1], block_hidden)
            inner_blocks = T.ceildiv(Mid.local_shape[1], block_inner)
            for bm in T.serial(T.ceildiv(X.local_shape[0], block_seq)):
                for bn in T.serial(inner_blocks):
                    T.clear(mid_acc)
                    for bk in T.Pipelined(hidden_blocks, num_stages=num_stages):
                        T.copy(
                            X[bm * block_seq : (bm + 1) * block_seq, bk * block_hidden : (bk + 1) * block_hidden],
                            lhs_local,
                        )
                        T.copy(
                            WUp[bk * block_hidden : (bk + 1) * block_hidden, bn * block_inner : (bn + 1) * block_inner],
                            up_local,
                        )
                        T.comm.all_gather(lhs_local, lhs_shared, direction="horizontal", axis=-1)
                        T.comm.all_gather(up_local, up_shared, direction="vertical", axis=0)
                        T.gemm(lhs_shared, up_shared, mid_acc)
                    for i, j in T.Tiles(mid_tile, parallel=True):
                        mid_tile[i, j] = T.Cast(dtype, T.max(mid_acc[i, j], T.float32(0)))
                    T.copy(mid_tile, Mid[bm * block_seq, bn * block_inner])

                for bh in T.serial(hidden_blocks):
                    T.clear(out_acc)
                    for bn in T.Pipelined(inner_blocks, num_stages=num_stages):
                        T.copy(
                            Mid[bm * block_seq : (bm + 1) * block_seq, bn * block_inner : (bn + 1) * block_inner],
                            mid_local,
                        )
                        T.copy(
                            WDown[bn * block_inner : (bn + 1) * block_inner, bh * block_hidden : (bh + 1) * block_hidden],
                            down_local,
                        )
                        T.comm.all_gather(mid_local, mid_shared, direction="horizontal", axis=-1)
                        T.comm.all_gather(down_local, down_shared, direction="vertical", axis=0)
                        T.gemm(mid_shared, down_shared, out_acc)
                    T.copy(out_acc, Y[bm * block_seq, bh * block_hidden])

    return main


def _lower_ffn():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        func = mesh_ffn_new(num_stages=2).with_attr("global_symbol", "main")
        mod = tir.transform.BindTarget(target)(tvm.IRModule.from_expr(func))
        mod = tl.transform.ResolveSunmmioMeshSymbols()(mod)
        if should_force_let_inline():
            mod = tl.transform.LetInline()(mod)
        for pipeline_pass in (
            tl.transform.LegalizeNegativeIndex(),
            tl.transform.InjectAssumes(),
            tl.transform.Simplify(),
            tl.transform.InferSramScope(),
            tl.transform.LegalizeSunmmioDataPath(),
            tl.transform.SunmmioLayoutInference(),
            tl.transform.LegalizeSunmmioGemm(),
            tl.transform.LowerTileOp(),
            tl.transform.LegalizeTilesLoop(),
            tl.transform.TilesLoop(),
            tl.transform.LegalizeVectorizedLoop(),
            tl.transform.LegalizeSafeMemoryAccess(),
            tl.transform.LowerAccessPtr(),
            tl.transform.Simplify(),
            tl.transform.HoistNonRestrictParams(),
            tl.transform.HoistBlockAnnotationsToFuncAttrs(),
        ):
            mod = pipeline_pass(mod)
    return tl.transform.IfStmtBinding()(mod)


def _pipeline_loops(stmt):
    loops = []

    def visit(node):
        if node is None:
            return
        if isinstance(node, tir.For):
            if node.annotations and "tl.sunmmio.pipeline.requested" in node.annotations:
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


def test_greedy_ffn_models_collective_order_and_relative_bank_precolors(monkeypatch, tmp_path):
    graph_path = tmp_path / "greedy_ffn_graph.json"
    monkeypatch.setenv("TL_SUNMMIO_PIPELINE_GRAPH_JSON", str(graph_path))
    planned = tl.transform.SunmmioPipelinePlanning(debug=False)(_lower_ffn())
    loops = _pipeline_loops(planned["main"].body)
    assert len(loops) == 2

    collective_ids = (2, 3, 5)
    collective_rank = {command_id: rank for rank, command_id in enumerate(collective_ids)}
    for loop in loops:
        annotations = loop.annotations
        assert bool(annotations["tl.sunmmio.pipeline.applied"])

        body_orders = [tuple(map(int, str(order).split("-"))) for order in annotations["body_orders"]]
        body_positions = {order: position for position, order in enumerate(body_orders)}
        # Commands 0-5 (ODMA1), 1-0 (ODMA0), and 0-4 (TensorCore) all
        # start at time zero.  Async launches must be emitted before blocking MMA.
        assert body_positions[(0, 5)] < body_positions[(0, 4)]
        assert body_positions[(1, 0)] < body_positions[(0, 4)]

        for name in ("prologue_orders", "body_orders", "epilogue_orders"):
            collective_orders = [
                tuple(map(int, str(order).split("-"))) for order in annotations[name] if int(str(order).split("-")[1]) in collective_ids
            ]
            assert collective_orders == sorted(
                collective_orders,
                key=lambda order: (order[0], collective_rank[order[1]]),
            )

        phase_maps = [
            {int(command_id): int(phase) for command_id, phase in phases.items()}
            for phases in annotations["runtime_bank_writer_phases"].values()
        ]
        striped_writers = next(phases for phases in phase_maps if 2 in phases and 5 in phases)
        assert striped_writers[2] != striped_writers[5]

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    resources = {command["id"]: command["resource"] for command in graph["commands"]}
    assert {command_id: resources[command_id] for command_id in collective_ids} == {2: 3, 3: 2, 5: 3}
    assert [(edge["source"], edge["target"], edge["distance"]) for edge in graph["edges"] if edge["kind"] == "collective_order"] == [
        (2, 3, 0),
        (3, 5, 0),
        (5, 2, 1),
    ]

    injected = tl.transform.InjectSunmmioPipeline()(planned)
    broadcasts = []
    allocated_shapes = {}

    def collect_injected(node):
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.broadcast_":
            broadcasts.append(node)
        if isinstance(node, tir.Block):
            for buffer in node.alloc_buffers:
                allocated_shapes[str(buffer.name)] = tuple(int(dim) for dim in buffer.shape)

    tir.stmt_functor.post_order_visit(
        injected["main"].body,
        collect_injected,
    )
    assert len(broadcasts) >= 4
    assert allocated_shapes["lhs_local"] == (2, 32, 32)
    assert allocated_shapes["up_local"] == (2, 32, 32)
    assert allocated_shapes["mid_local"] == (2, 32, 32)
    assert allocated_shapes["down_local"] == (2, 32, 32)
