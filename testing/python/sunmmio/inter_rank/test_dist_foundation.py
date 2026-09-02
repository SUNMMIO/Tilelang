"""基础 Rank/world 语法与 MeshTensor Rank placement 测试。"""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


@tilelang.jit(target="sunmmio")
def rank_sharded_kernel_factory(M, N, world_size: int = 1):
    @T.prim_func
    def main(
        A: T.MeshTensor(
            (M, N),
            placement=T.placement.row_shard(0),
            device_mesh_config=(2, 2),
            dtype=T.bfloat16,
            rank_placement=T.dist.placement.shard(0),
        ),
        rank_id: T.dist.RankId,
    ):
        with T.Kernel() as core_id:
            world = T.dist.world_size()
            valid_m, valid_n = A.get_local_extent(core_id, rank_id=rank_id)
            if rank_id < world:
                T.evaluate(valid_m + valid_n)

    return main


@tilelang.jit(target="sunmmio")
def replicated_kernel_factory(M, N, world_size: int = 1):
    @T.prim_func
    def main(
        A: T.MeshTensor(
            (M, N),
            placement=T.placement.replicated(),
            device_mesh_config=(2, 2),
            dtype=T.bfloat16,
        ),
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            T.evaluate(rank_id + T.dist.world_size())

    return main


def _as_int_tuple(values):
    return tuple(int(value) for value in values)


def test_world_size_rank_id_and_rank_sharding_metadata():
    func = rank_sharded_kernel_factory.get_tir(17, 32, world_size=4)

    assert int(func.attrs["tl.dist.world_size"]) == 4
    rank_id_param_index = int(func.attrs["tl.dist.rank_id_param_index"])
    assert rank_id_param_index == 1
    assert str(func.params[rank_id_param_index].dtype) == "int32"

    meta = func.attrs["tensor_meta"]["A"]
    assert _as_int_tuple(meta["global_shape"]) == (17, 32)
    assert _as_int_tuple(meta["rank_shape"]) == (5, 32)
    assert _as_int_tuple(meta["local_shape"]) == (3, 32)
    assert _as_int_tuple(meta["rank_placement"]) == (1, 0)
    assert int(meta["world_size"]) == 4
    assert _as_int_tuple(func.buffer_map[func.params[0]].shape) == (3, 32)


def test_rank_sharding_non_divisible_valid_extents():
    func = rank_sharded_kernel_factory.get_tir(17, 32, world_size=4)
    meta = func.attrs["tensor_meta"]["A"]

    assert [int(T.get_rank_extent(meta, rank_id)[0]) for rank_id in range(4)] == [5, 4, 4, 4]
    assert int(T.get_local_extent(meta, 0, rank_id=0)[0]) == 3
    assert int(T.get_local_extent(meta, 2, rank_id=0)[0]) == 2
    assert int(T.get_local_extent(meta, 0, rank_id=1)[0]) == 2
    assert int(T.get_local_extent(meta, 3, rank_id=3)[0]) == 2


def test_omitted_rank_placement_is_replicated():
    func = replicated_kernel_factory.get_tir(17, 32, world_size=4)
    meta = func.attrs["tensor_meta"]["A"]

    assert _as_int_tuple(meta["global_shape"]) == (17, 32)
    assert _as_int_tuple(meta["rank_shape"]) == (17, 32)
    assert _as_int_tuple(meta["local_shape"]) == (17, 32)
    assert _as_int_tuple(meta["rank_placement"]) == (0, -1)
    assert _as_int_tuple(T.get_rank_extent(meta, 3)) == (17, 32)
    assert _as_int_tuple(T.get_local_extent(meta, 3)) == (17, 32)


@pytest.mark.parametrize("world_size", [0, -1, True, 2.0, None])
def test_world_size_must_be_a_positive_int(world_size):
    with pytest.raises(ValueError, match="world_size must be a positive int"):
        rank_sharded_kernel_factory.get_tir(17, 32, world_size=world_size)


def test_rank_placement_defaults_to_world_size_one():
    tensor = T.MeshTensor(
        (16, 32),
        placement=T.placement.replicated(),
        device_mesh_config=(2, 2),
        dtype=T.bfloat16,
        rank_placement=T.dist.placement.shard(0),
    )
    assert tensor.rank_shape == (16, 32)
    assert tensor.meta_data["world_size"] == 1


def test_rank_placement_rejects_invalid_dimension():
    with pytest.raises(ValueError, match="Rank shard dim must be a non-negative int"):
        T.dist.placement.shard(-1)


def test_rank_sharded_local_extent_requires_rank_id():
    func = rank_sharded_kernel_factory.get_tir(17, 32, world_size=4)
    meta = func.attrs["tensor_meta"]["A"]
    with pytest.raises(ValueError, match="rank_id is required"):
        T.get_local_extent(meta, 0)


def test_prim_func_rejects_multiple_rank_id_parameters():
    with pytest.raises(ValueError, match="at most one T.dist.RankId"):

        @T.prim_func
        def invalid(rank_id: T.dist.RankId, other_rank_id: T.dist.RankId):
            T.evaluate(rank_id + other_rank_id)


def test_world_size_returns_compile_time_intimm():
    func = replicated_kernel_factory.get_tir(8, 16, world_size=3)
    int_imms = []

    def visit(node):
        if isinstance(node, tvm.tir.IntImm):
            int_imms.append(int(node))

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    assert 3 in int_imms


def test_world_size_partitions_the_jit_cache_and_rank_shape():
    world_two = rank_sharded_kernel_factory.get_tir(17, 32, world_size=2)
    world_four = rank_sharded_kernel_factory.get_tir(17, 32, world_size=4)

    assert int(world_two.attrs["tl.dist.world_size"]) == 2
    assert int(world_four.attrs["tl.dist.world_size"]) == 4
    assert _as_int_tuple(world_two.attrs["tensor_meta"]["A"]["rank_shape"]) == (9, 32)
    assert _as_int_tuple(world_four.attrs["tensor_meta"]["A"]["rank_shape"]) == (5, 32)


def test_world_size_builtin_defaults_to_one_without_compile_context():
    assert int(T.dist.world_size()) == 1


def test_world_and_rank_id_metadata_survive_to_device_tir():
    func = rank_sharded_kernel_factory.get_tir(17, 32, world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.MakePackedAPI")
    device_funcs = [candidate for candidate in result.device_mod.functions.values() if isinstance(candidate, tvm.tir.PrimFunc)]

    assert len(device_funcs) == 1
    device_func = device_funcs[0]
    assert int(device_func.attrs["tl.dist.world_size"]) == 4
    rank_id_param_index = int(device_func.attrs["tl.dist.rank_id_param_index"])
    assert rank_id_param_index == 0
    assert device_func.params[rank_id_param_index].name == "rank_id"
    assert str(device_func.params[rank_id_param_index].dtype) == "int32"

    packed_script = result.pass_snapshot("tl.MakePackedAPI").mod.script()
    assert '"shape[0]", T.int64(5)' in packed_script
    assert '"shape[0]", T.int64(17)' not in packed_script
