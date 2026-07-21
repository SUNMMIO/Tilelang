import tilelang
import tilelang.utils.target as _target_utils
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalizeSunmmio
from tilelang.utils.target import determine_target


def test_lower_and_legalize_skips_region_validator_when_disabled(monkeypatch):
    monkeypatch.setattr(_target_utils, "ENABLE_SUNMMIO_REGION_VALIDATION", False)

    def forbidden_validator():
        raise AssertionError("ValidateTileViewRegions must not be constructed when validation is disabled")

    monkeypatch.setattr(tilelang.transform, "ValidateTileViewRegions", forbidden_validator)

    @T.prim_func
    def main(A: T.Tensor((32, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel():
            A_shared = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            T.copy(A, A_shared)
            T.copy(A_shared, B)

    target = determine_target("Sunmmio", return_object=True)
    with tvm.target.Target(target):
        lowered = LowerAndLegalizeSunmmio(tvm.IRModule({"main": main}), target)

    assert lowered.get_global_vars()
