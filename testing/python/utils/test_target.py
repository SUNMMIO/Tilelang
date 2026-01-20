import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.carver.arch import driver
from tilelang.engine.lower import canon_target_host
from tilelang.utils.target import determine_target, \
    target_is_sunmmio


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


def test_sunmmio_target_binding():
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    print("Device mesh config:", device_mesh_config)

    def example_tensor_annot(shape):
        MyTensor = T.MeshTensor(
            shape, T.MeshShardingPolicy(y=0, x=1), device_mesh_config, dtype="float32")

        @T.prim_func
        def kernel(A: MyTensor):
            pass

        return kernel

    func = example_tensor_annot((
        1024,
        1024,
    ))
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
    for key in ["device_mesh_nrow", "device_mesh_nrow"]:
        assert any(key in mattr for mattr in target_attr.attrs['mattr'])
