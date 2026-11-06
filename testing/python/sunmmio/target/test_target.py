import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.carver.arch import driver
from tilelang.engine.lower import canon_target_host
from tilelang.utils.target import (
    determine_target,
    get_sunmmio_target_mesh_config,
    target_context,
    target_is_sunmmio,
)


def test_sunmmio_target():
    target_name = "Sunmmio"
    target = determine_target(target_name, return_object=True)
    assert target.attrs["mcpu"] == "sunmmio-a4e"
    assert target_is_sunmmio(target)

    target_host = canon_target_host(target, None)

    target_host = tvm.target.Target.canon_target(target_host)
    target = tvm.target.Target(target, target_host)

    print(target)
    print(target.attrs)


def test_sunmmio_mesh_config_comes_from_target():
    assert get_sunmmio_target_mesh_config("sunmmio") == (4, 4)

    for nrows, ncols in ((3, 4), (4, 3)):
        target = determine_target(
            f"llvm -mcpu=sunmmio-a4e -mattr=device_mesh_nrow_{nrows},device_mesh_ncol_{ncols}",
            return_object=True,
        )
        assert get_sunmmio_target_mesh_config(target) == (nrows, ncols)
        with target:
            assert get_sunmmio_target_mesh_config() == (nrows, ncols)


def test_sunmmio_device_properties_do_not_duplicate_mesh_config():
    properties = driver.get_sunmmio_device_properties()

    assert not hasattr(properties, "mesh_config")
    assert properties.RsramPerCore == 1536000
    assert properties.WSRAMperCore == 2097152
    assert properties.ASRAMPerCore == 1048576


def test_target_context_decorator():
    @target_context("Sunmmio")
    def get_current_target():
        target = tvm.target.Target.current()
        assert target_is_sunmmio(target)
        return target.attrs["mcpu"]

    assert get_current_target() == "sunmmio-a4e"
    assert get_current_target.__name__ == "get_current_target"


def test_sunmmio_target_binding():
    def example_tensor_annot(shape):
        MyTensor = T.MeshTensor(shape, T.placement.full_shard(0, 1), dtype="float32")

        @T.prim_func
        def kernel(A: MyTensor):
            pass

        return kernel

    func = example_tensor_annot(
        (
            1024,
            1024,
        )
    )
    print(func)
    target_name = "Sunmmio"
    target = determine_target(target_name, return_object=True)

    mod = func
    if isinstance(func, tvm.tir.PrimFunc):
        mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    mod = tvm.tir.transform.BindTarget(target)(mod)
    prim_func = mod["kernel"]
    prim_func_attr = prim_func.attrs
    target_attr = prim_func_attr["target"]
    assert target_attr.attrs["mcpu"] == "sunmmio-a4e"
    for key in ["device_mesh_nrow", "device_mesh_ncol"]:
        assert any(key in mattr for mattr in target_attr.attrs["mattr"])
