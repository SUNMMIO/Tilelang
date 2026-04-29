from .gemm_base import GemmBase
from tilelang.layout import make_zz_layout, make_zn_layout
from tilelang import tvm as tvm
import tvm_ffi
from tvm.target import Target
from tvm import tir
from tilelang.transform.simplify import _Simplify
from tilelang import language as T
from tilelang.utils.language import (
    retrieve_shape,
)
from tilelang.language.utils import (
    buffer_region_to_tile_region,
)

_get_sunmmio_layout_block_shape = tvm_ffi.get_global_func("tl.target.GetSunmmioLayoutBlockShape")


def _sunmmio_block_shape(dtype):
    """Dtype-dependent block shape for Sunmmio A4E Tensor Core.

    Delegates to the C++ GetSunmmioLayoutBlockShape via FFI so that the
    bit-width → block-shape mapping is defined in exactly one place.
    """
    target = Target.current()
    return list(_get_sunmmio_layout_block_shape(target, tvm.DataType(dtype)))


class GemmSunmmio(GemmBase):
    def infer_layout(self, target: Target, thread_nums: int):
        if self.is_gemm_sunmmio_scope():
            # A (ASRAM): ZZ with dtype-dependent block shape
            a_block = _sunmmio_block_shape(self.A.dtype)
            a_layout = make_zz_layout(self.A, block_shape=a_block)
            # B (WSRAM): ZZ if transB (TMM.MT mode), ZN if !transB (TMM.MN mode)
            b_block = _sunmmio_block_shape(self.B.dtype)
            rank_b = len(self.B.shape)
            axes_b = [rank_b - 2, rank_b - 1]
            if self.trans_B:
                b_layout = make_zz_layout(self.B, block_shape=b_block)
            else:
                b_layout = make_zn_layout(self.B.shape, axes_b, b_block)
            # C (RSRAM): ZZ with block shape from GetSunmmioLayoutBlockShape
            c_block = _sunmmio_block_shape(self.C.dtype)
            c_layout = make_zz_layout(self.C, block_shape=c_block)
            return {
                self.A: a_layout,
                self.B: b_layout,
                self.C: c_layout,
            }
        else:
            raise ValueError(f"Unsupported gemm combination of Sunmmio, A: {self.A.scope()}, B: {self.B.scope()}, C: {self.C.scope()}")

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
            raise ValueError(f"Unsupported gemm combination of Sunmmio, A: {self.A.scope()}, B: {self.B.scope()}, C: {self.C.scope()}")

    def is_gemm_sunmmio_scope(self) -> bool:
        a_check = self.A.scope() == "shared.asram"
        b_check = self.B.scope() == "shared.wsram"
        c_check = self.C.scope() == "shared.rsram"
        return a_check and b_check and c_check
