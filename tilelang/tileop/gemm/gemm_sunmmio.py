from .gemm_base import GemmBase
from tilelang.layout import make_blockwise_zz_layout
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
        if self.is_gemm_sunmmio_scope():
            return {
                self.A: make_blockwise_zz_layout(self.A),
                self.B: make_blockwise_zz_layout(self.B),
                self.C: make_blockwise_zz_layout(self.C),
            }
        else:
            raise ValueError(
                f"Unsupported gemm combination of Sunmmio, A: {self.A.scope()}, B: {self.B.scope()}, C: {self.C.scope()}"
            )

    def lower(self, layout_map: dict, target: Target, thread_nums: int, thread_var: tir.Var):

        if self.is_gemm_sunmmio_scope():
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
                f"Unsupported gemm combination of Sunmmio, A: {self.A.scope()}, B: {self.B.scope()}, C: {self.C.scope()}"
            )

    def is_gemm_sunmmio_scope(self) -> bool:
        a_check = self.A.scope() == 'shared.asram'
        b_check = self.B.scope() == 'shared.wsram'
        c_check = self.C.scope() == 'shared.rsram'
        return a_check and b_check and c_check
