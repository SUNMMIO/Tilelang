from .gemm_base import GemmBase
from tilelang.layout import make_blockwise_zz_layout
from tilelang.utils.language import is_shared
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tir
from tilelang.transform.simplify import _Simplify
from tilelang import language as T
from tilelang.utils.language import (
    retrieve_shape,)
from tilelang.language.utils import (
    buffer_region_to_tile_region,)


class GemmSunmmio(GemmBase):

    def infer_layout(self, target: Target, thread_nums: int):
        assert self.A.scope() == 'shared.asram'
        assert self.B.scope() == 'shared.wsram'
        assert self.C.scope() == 'shared.rsram'
        if self.is_gemm_sss():
            return {
                self.A: make_blockwise_zz_layout(self.A),
                self.B: make_blockwise_zz_layout(self.B),
                self.C: make_blockwise_zz_layout(self.C),
            }
        else:
            raise ValueError(
                f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}, C: {self.C.scope()}"
            )

    def lower(self, layout_map: dict, target: Target, thread_nums: int, thread_var: tir.Var):
        assert self.A.scope() == 'shared.asram'
        assert self.B.scope() == 'shared.wsram'
        assert self.C.scope() == 'shared.rsram'
        if self.is_gemm_sss():
            A_shape = retrieve_shape(self.ARegion)
            B_shape = retrieve_shape(self.BRegion)
            C_shape = retrieve_shape(self.CRegion)
            A_arg = buffer_region_to_tile_region(self.ARegion, "r", [r for r in A_shape])
            B_arg = buffer_region_to_tile_region(self.BRegion, "r", [r for r in B_shape])
            C_arg = buffer_region_to_tile_region(self.CRegion, "rw", [r for r in C_shape])

            args = [A_arg, B_arg, C_arg, self.trans_A, self.trans_B, self.clear_accum]

            @T.prim_func
            def _gemm_sss() -> None:
                tir.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.mma_sunmmio"),
                    *args,
                )

            return _Simplify(_gemm_sss, inline_let=True)

        else:
            raise ValueError(
                f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}, C: {self.C.scope()}"
            )

    def is_gemm_sss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B) and is_shared(self.C)
