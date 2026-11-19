"""Bounded FP32 scan: preserve tile SSA and real DMA dependencies."""

import pytest
import tilelang.language as T
from tilelang.layout import make_aligned_row_major
from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt


def kernel(n=512, inplace=True, dtype="float32", reverse=False, sliced=False, rank2=False):
    placement = T.placement.replicated()
    layout = make_aligned_row_major((1, n), dtype, align_bytes=1024)
    shape = (1, n) if rank2 else (n,)

    @T.prim_func
    def main(source: T.MeshTensor((1, n), placement, dtype, layout=layout), output: T.MeshTensor((1, n), placement, dtype, layout=layout)):
        with T.Kernel(T.mesh_ncores()):
            a = T.alloc_shared(shape, dtype, scope="shared.rsram")
            if not inplace:
                b = T.alloc_shared(shape, dtype, scope="shared.rsram")
            T.copy(source[0, 0], a)
            if sliced:
                T.cumsum(a[128:256], dim=0)
            elif inplace:
                T.cumsum(a, dim=0, reverse=reverse)
            else:
                T.cumsum(a, b, dim=0, reverse=reverse)
            if inplace:
                T.copy(a, output[0, 0])
            else:
                T.copy(b, output[0, 0])

    return main


@pytest.mark.parametrize("n,inplace", [(256, True), (512, True), (1024, False), (4096, True)])
def test_scan_ssa_and_dma_boundaries(tmp_path, n, inplace):
    src = validate_sunmmio_codegen_with_npuir_opt(kernel(n, inplace), tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
    assert src.count("suvm.tile.extract_slice") == 7 * (n // 128)
    assert src.count("suvm.tile.insert_slice") == 7 * (n // 128)
    assert src.count("suvm.tile.load") == n // 128
    assert src.count("suvm.tile.store") == n // 128
    first_copy = src.index("suvm.copy_async")
    first_load = src.index("suvm.tile.load")
    assert "suvm.sync odma0" in src[first_copy:first_load]
    final_store = src.rindex("suvm.tile.store")
    final_copy = src.rindex("suvm.copy_async")
    assert "suvm.sync vector,rsram" in src[final_store:final_copy]
    # One chain per segment: no memory operation or completion wait between
    # its first extract and the carry pick.
    for chain in src.split("suvm.tile.load")[1:]:
        stages = chain[chain.index("suvm.tile.extract_slice") : chain.index("suvm.tile.pick")]
        assert stages.count("suvm.tile.addf") == 8
        assert "suvm.tile.store" not in stages and "suvm.sync" not in stages


def test_minimum_segment_with_full_dram_carrier(tmp_path):
    placement = T.placement.replicated()
    layout = make_aligned_row_major((256,), "float32", align_bytes=1024)

    @T.prim_func
    def main(output: T.MeshTensor((256,), placement, T.float32, layout=layout)):
        with T.Kernel(T.mesh_ncores()):
            local = T.alloc_shared((128,), T.float32, scope="shared.rsram")
            carrier = T.alloc_shared((256,), T.float32, scope="shared.rsram")
            T.fill(local, T.float32(1))
            T.cumsum(local)
            T.fill(carrier, T.float32(0))
            for i in T.Tiles([128]):
                carrier[i] = local[i]
            T.copy(carrier, output)

    src = validate_sunmmio_codegen_with_npuir_opt(main, tmp_path, opt_args=("--verify-each", "--suvm-to-llvm-pipeline"))
    assert src.count("suvm.tile.extract_slice") == 7
    assert src.count("suvm.tile.insert_slice") == 7
    assert "suvm.transform_layout_async" not in src


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"reverse": True}, "reverse scan"),
        ({"dtype": "bfloat16"}, "FP32 only"),
        ({"n": 130}, "divisible by 128"),
        ({"n": 4224}, "divisible by 128"),
        ({"sliced": True}, "complete buffer regions"),
        ({"rank2": True}, "rank-1 buffers"),
    ],
)
def test_unsupported_scan_is_rejected(tmp_path, kwargs, message):
    with pytest.raises(Exception, match=message):
        validate_sunmmio_codegen_with_npuir_opt(kernel(**kwargs), tmp_path)
