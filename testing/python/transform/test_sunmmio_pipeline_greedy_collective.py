import json

import tilelang as tl
from tilelang import tvm
from tilelang.engine.phase import should_force_let_inline
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from testing.python.transform.sunmmio_mesh_kernel_new_syntax_reference import mesh_ffn_new
from tvm import tir


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
