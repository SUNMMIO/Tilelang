from .gemm_base import GemmBase
from tilelang.layout import make_blockwise_zz_layout
from tilelang.utils.language import is_shared
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tir
from tilelang.transform.simplify import _Simplify
from tilelang import language as T
from tilelang.utils.language import retrieve_ptr


class GemmSunmmio(GemmBase):

    def infer_layout(self, target: Target, thread_nums: int):
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
        if self.is_gemm_sss():
            args = []

            def add_info(args, region):
                args.append(len(region.buffer.shape))
                for it in region.region:
                    args.append(it.extent)
                args.append(region.buffer.dtype)
                layout = layout_map[region.buffer]
                for it in layout.input_size:
                    args.append(it)
                for it in layout.forward_index:
                    args.append(it)
                args.append(region.buffer.scope())
                for it in region.region:
                    args.append(it.min)
                if region != self.CRegion:
                    args.append(retrieve_ptr(region.buffer, access_type="r"))
                else:
                    args.append(retrieve_ptr(region.buffer, access_type="w"))

            add_info(args, self.ARegion)
            add_info(args, self.BRegion)
            add_info(args, self.CRegion)

            args.append(self.trans_A)
            args.append(self.trans_B)
            args.append(self.clear_accum)

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
