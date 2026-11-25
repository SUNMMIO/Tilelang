"""Measure LowerDistSync on saved public-kernel pass inputs (outside pytest).

Run with --capture DIR on the baseline checkout, then --replay DIR on
both checkouts. Each sample uses the same pre-pass IR, without timing
frontend construction, MLIR codegen, or the A4E verifier.
"""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import tilelang
import tilelang.language as T
from tilelang import tvm

from testing.python.sunmmio.common.codegen_validation import codegen_sunmmio_suvm_mlir, validate_suvm_mlir_with_npuir_opt
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


@tilelang.jit(target="sunmmio")
def many_writes_kernel(count, payload, world_size: int = 1):
    @T.prim_func
    def main(B: T.MeshTensor((count * 128,), T.placement.replicated(), T.int32), rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((count * 128,), T.int32)
            dst = T.alloc_shared((32,), T.int32)
            s = T.dist.signal(expect="manual")
            for i in T.unroll(count, explicit=True):
                src[i * 128] = 1
            T.copy(src, B)
            if payload:
                T.dist.put(src[0:32], dst, (rank_id + 1) % world_size, signal=s, submit=True)
            else:
                T.dist.put_signal(s, (rank_id + 1) % world_size, submit=True)

    return main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture", type=Path)
    source.add_argument("--replay", type=Path)
    args = parser.parse_args()
    root = args.capture or args.replay
    root.mkdir(parents=True, exist_ok=True)
    tilelang.env.disable_cache()
    for payload in (False, True):
        for n in (128, 256, 512):
            name = f"{'payload' if payload else 'notify'}-{n}"
            p = root / (name + ".json")
            if args.capture:
                kernel = many_writes_kernel.get_tir(count=n, payload=payload, world_size=2)
                r = lower_to_device_tir(kernel, capture_before_passes="tl.LowerDistSync")
                p.write_text(tvm.ir.save_json(r.pass_snapshot("tl.LowerDistSync", when="before").mod))
                case = root / name
                case.mkdir(exist_ok=True)
                validate_suvm_mlir_with_npuir_opt(
                    codegen_sunmmio_suvm_mlir(r.device_mod), case, opt_args=("--verify-each", "--suvm-device-validate")
                )
            mod = tvm.ir.load_json(p.read_text())
            run = tilelang.transform.LowerDistSync()
            times = []
            for _ in range(3):
                start = perf_counter()
                run(mod)
                times.append(1000 * (perf_counter() - start))
            print(f"{name}: {median(times):.2f} ms", flush=True)


if __name__ == "__main__":
    main()
