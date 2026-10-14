"""Shared TIR inspection and single-stage lowering helpers for dist tests."""

import tilelang
from tilelang import tvm
from tilelang.utils.target import determine_target


def prim_funcs(func_or_mod):
    """Return the TIR functions from either a PrimFunc or an IRModule."""
    funcs = func_or_mod.functions.values() if isinstance(func_or_mod, tvm.IRModule) else (func_or_mod,)
    return [func for func in funcs if isinstance(func, tvm.tir.PrimFunc)]


def single_prim_func(func_or_mod):
    funcs = prim_funcs(func_or_mod)
    assert len(funcs) == 1
    return funcs[0]


def collect_nodes(func_or_mod, node_type, predicate=lambda node: True):
    """Collect nodes in post-order; ordering assertions must also inspect control flow."""
    nodes = []

    def visit(node):
        if isinstance(node, node_type) and predicate(node):
            nodes.append(node)

    for func in prim_funcs(func_or_mod):
        tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return nodes


def collect_calls(func_or_mod, op_name=None):
    return collect_nodes(
        func_or_mod,
        tvm.tir.Call,
        lambda call: isinstance(call.op, tvm.ir.Op) and (op_name is None or call.op.name == op_name),
    )


def collect_op_names(func_or_mod):
    return [call.op.name for call in collect_calls(func_or_mod)]


def collect_op_arg_counts(func_or_mod, op_name):
    return [len(call.args) for call in collect_calls(func_or_mod, op_name)]


def collect_dist_put_kinds(func_or_mod):
    return [str(call.args[3].value) for call in collect_calls(func_or_mod, "tl.dist_put_")]


def signal_counts(func):
    return {str(kind): int(count) for kind, count in func.attrs["tl.dist.signal_counts"].items()}


def base_mod(func):
    """Bind the test target and resolve mesh symbols for isolated dist pass tests."""
    target = tvm.target.Target(determine_target("sunmmio", return_object=True))
    mod = tvm.IRModule({"main": func.with_attr("target", target)})
    return tilelang.transform.ResolveSunmmioMeshSymbols()(mod)


def lower_collectives(func):
    return tilelang.transform.LowerDistCollectives()(base_mod(func))
