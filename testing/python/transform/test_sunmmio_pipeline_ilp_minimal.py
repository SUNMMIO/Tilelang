"""Minimal three-operation reproducer for the SunMMIO ILP pipeline."""

import tilelang.language as T
from tilelang.language.mesh_tensor import MeshReplicationType
from tilelang.layout import make_zz_layout
from testing.python.sunmmio.common.compile_pipeline import target

from testing.python.transform.test_tilelang_transform_sunmmio_pipeline_strict_ilptest import (
    _annotation_buffer_names,
    _build_pipeline_modules,
    _extract_pipeline_annotations,
)


@target("Sunmmio")
def minimal_three_op_gemm(
    M=512,
    N=512,
    K=256,
    block_M=128,
    block_N=128,
    block_K=32,
    num_stages=2,
):
    a_policy = T.MeshShardingPolicy(y=0, replicate=MeshReplicationType.ROW)
    b_policy = T.MeshShardingPolicy(x=1, replicate=MeshReplicationType.COLUMN)
    c_policy = T.MeshShardingPolicy(y=0, x=1)

    a_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    b_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    c_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), a_policy, (4, 4), T.bfloat16, layout=a_layout),
        B: T.MeshTensor((K, N), b_policy, (4, 4), T.bfloat16, layout=b_layout),
        C: T.MeshTensor((M, N), c_policy, (4, 4), T.float32, layout=c_layout),
    ):
        with T.Kernel() as _:
            local_M, local_K = A.local_shape
            local_N = B.local_shape[1]
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B_shared = T.alloc_shared((block_K, block_N), T.bfloat16)
            C_shared = T.alloc_shared((block_M, block_N), T.float32)

            for by in T.serial(T.ceildiv(local_M, block_M)):
                for bx in T.serial(T.ceildiv(local_N, block_N)):
                    T.clear(C_shared)
                    for k in T.Pipelined(T.ceildiv(local_K, block_K), num_stages=num_stages):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_shared)
                    T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def test_minimal_three_op_pipeline_ilp_translation():
    def factory():
        return minimal_three_op_gemm()

    factory._strict_case_name = "minimal_three_op_gemm"
    factory._requested_num_stages = 2

    planned, injected = _build_pipeline_modules(factory)
    annotations = _extract_pipeline_annotations(planned["main"].body)

    assert annotations is not None
    # The three source-level operations become six schedulable nodes after
    # SunMMIO datapath legalization: A has two write/consume flows, while B
    # has one. Keep this explicit because that split is the ping/pong case
    # this reproducer is intended to isolate.
    assert len(annotations["body_orders"]) == 6
    boundary_orders = [
        *annotations["prologue_orders"],
        *annotations["epilogue_orders"],
    ]
    assert len(boundary_orders) == 6
    assert {int(str(order).split("-")[1]) for order in boundary_orders} == set(range(6))
    assert _annotation_buffer_names(annotations, "runtime_banked_buffers") == [
        "A_shared",
        "B_shared",
    ]
    writer_phases = annotations["runtime_bank_writer_phases"]
    reader_phases = annotations["runtime_bank_reader_phases"]
    a_buffer = next(buffer for buffer in writer_phases if buffer.name == "A_shared")
    b_buffer = next(buffer for buffer in writer_phases if buffer.name == "B_shared")
    flip_modes = annotations["runtime_bank_flip_modes"]
    assert int(flip_modes[a_buffer]) == 0
    assert int(flip_modes[b_buffer]) == 1
    assert "A_shared" not in _annotation_buffer_names(annotations, "runtime_multiversion_buffers")
    a_writers = {int(op): int(phase) for op, phase in writer_phases[a_buffer].items()}
    a_readers = {int(op): int(phase) for op, phase in reader_phases[a_buffer].items()}
    b_writers = {int(op): int(phase) for op, phase in writer_phases[b_buffer].items()}
    b_readers = {int(op): int(phase) for op, phase in reader_phases[b_buffer].items()}
    assert a_writers.keys() == {1, 4}
    assert a_readers.keys() == {3, 5}
    assert a_writers[1] == a_readers[3]
    assert a_writers[4] == a_readers[5]
    assert a_writers[1] != a_writers[4]
    assert b_writers.keys() == {2}
    assert b_readers.keys() == {3, 5}
    assert b_writers[2] == b_readers[3] == b_readers[5]

    script = injected.script(show_meta=True)
    assert "A_shared_ping = T.Buffer((128, 32)" in script
    assert "A_shared_pong = T.Buffer((128, 32)" in script
    assert "A_shared_ping" in script and "A_shared_pong" in script
    assert "B_shared_ping" in script and "B_shared_pong" in script
    assert "T.mma_sunmmio" in script
