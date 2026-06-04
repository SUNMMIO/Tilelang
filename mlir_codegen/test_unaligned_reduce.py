"""
Generate final SunMMIO TIR for unaligned reduce cases.

This script intentionally stops before SunMMIO MLIR codegen.  It uses the
mlir_codegen compile pipeline to run TileLang/TIR passes, then writes the final
device TIR for inspection.

Examples:

    python3 mlir_codegen/test_unaligned_reduce.py
    python3 mlir_codegen/test_unaligned_reduce.py --case 0 --echo
    python3 mlir_codegen/test_unaligned_reduce.py --case 5x43x249_axis2_clear_false
"""

import argparse
import os
from dataclasses import dataclass

import tilelang
import tilelang.language as T
from compile_pipeline import compile_test

tilelang.env.disable_cache()

if not hasattr(tilelang.transform, "SunmmioPipelinePlanning"):

    def _sunmmio_pipeline_planning_compat(debug=False):
        return tilelang.transform.PipelinePlanning()

    tilelang.transform.SunmmioPipelinePlanning = _sunmmio_pipeline_planning_compat

if not hasattr(tilelang.transform, "InjectSunmmioPipeline"):
    tilelang.transform.InjectSunmmioPipeline = tilelang.transform.InjectSoftwarePipeline


@dataclass(frozen=True)
class ReduceCase:
    shape: tuple[int, ...]
    axis: int
    clear: bool

    @property
    def label(self) -> str:
        shape_label = "x".join(str(dim) for dim in self.shape)
        clear_label = "clear_true" if self.clear else "clear_false"
        return f"{shape_label}_axis{self.axis}_{clear_label}"


UNALIGNED_REDUCE_CASES = [
    ReduceCase((1000,), 0, True),
    ReduceCase((1000,), 0, False),
    ReduceCase((33, 50), 0, True),
    ReduceCase((33, 50), 0, False),
    ReduceCase((33, 50), 1, True),
    ReduceCase((33, 50), 1, False),
    ReduceCase((5, 43, 249), 0, True),
    ReduceCase((5, 43, 249), 0, False),
    ReduceCase((5, 43, 249), 1, True),
    ReduceCase((5, 43, 249), 1, False),
    ReduceCase((5, 43, 249), 2, True),
    ReduceCase((5, 43, 249), 2, False),
]


def parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in ("1", "true", "t", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def apply_reduce_op(reduce_op: str, buffer, out, reduce_axis: int, clear: bool) -> None:
    if reduce_op == "sum":
        T.reduce_sum(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "abssum":
        T.reduce_abssum(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "max":
        T.reduce_max(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "absmax":
        T.reduce_absmax(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "min":
        T.reduce_min(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "bitand":
        T.reduce_bitand(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "bitor":
        T.reduce_bitor(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "bitxor":
        T.reduce_bitxor(buffer, out, dim=reduce_axis, clear=clear)
    else:
        raise ValueError(f"Unsupported reduce op: {reduce_op!r}")


def reduce_kernel_builder(case: ReduceCase, dtype: str = "float16", reduce_op: str = "sum"):
    out_shape = list(case.shape[: case.axis]) + list(case.shape[case.axis + 1 :])
    if not out_shape:
        out_shape = [1]

    @T.prim_func
    def main(A: T.Tensor(case.shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(case.shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            if not case.clear:
                T.copy(Out, Out_shared)
            apply_reduce_op(reduce_op, A_shared, Out_shared, case.axis, case.clear)
            T.copy(Out_shared, Out)

    return main


def select_cases(selector: str) -> list[tuple[int, ReduceCase]]:
    selector = selector.strip()
    if selector == "all":
        return list(enumerate(UNALIGNED_REDUCE_CASES))

    cases_by_label = {case.label: (idx, case) for idx, case in enumerate(UNALIGNED_REDUCE_CASES)}
    selected: list[tuple[int, ReduceCase]] = []
    for item in selector.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            idx = int(item)
            selected.append((idx, UNALIGNED_REDUCE_CASES[idx]))
        elif item in cases_by_label:
            selected.append(cases_by_label[item])
        else:
            valid = ", ".join(case.label for case in UNALIGNED_REDUCE_CASES)
            raise ValueError(f"Unknown case {item!r}. Valid labels: {valid}")
    return selected


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")


def compile_case(idx: int, case: ReduceCase, args: argparse.Namespace) -> str:
    func = reduce_kernel_builder(case, dtype=args.dtype, reduce_op=args.reduce_op)
    log_dir = os.path.join(args.log_root, f"case{idx}_{case.label}_{args.reduce_op}_{args.dtype}")
    os.makedirs(log_dir, exist_ok=True)

    _, device_mod = compile_test(
        func,
        out_idx=[1],
        target="Sunmmio",
        log_pass_output=True,
        log_dir=log_dir,
        remove_header=True,
        log_passes=["DeviceMod"],
    )

    tir_text = device_mod.script(show_meta=False)
    tir_path = os.path.join(log_dir, "final_tir.py")
    write_text(tir_path, tir_text)

    if args.echo:
        print(f"\n'=== {case.label} {args.reduce_op} {args.dtype} ==='")
        print(tir_text, end="" if tir_text.endswith("\n") else "\n")

    print(f"Saved final TIR to {tir_path}")
    return tir_path


def parse_args() -> argparse.Namespace:
    default_log_root = os.environ.get(
        "TL_UNALIGNED_REDUCE_LOG_ROOT",
        os.path.join(os.path.dirname(__file__), "logs_unaligned_reduce"),
    )
    default_case = os.environ.get("TL_UNALIGNED_REDUCE_CASE", "all")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default=default_case,
        help="Case selector: all, comma-separated indexes, or comma-separated case labels.",
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("TL_UNALIGNED_REDUCE_DTYPE", "float16"),
        help="Input/output dtype.",
    )
    parser.add_argument(
        "--reduce-op",
        default=os.environ.get("TL_UNALIGNED_REDUCE_OP", "sum"),
        choices=("sum", "abssum", "max", "absmax", "min", "bitand", "bitor", "bitxor"),
        help="Reduce operator to lower.",
    )
    parser.add_argument(
        "--log-root",
        default=default_log_root,
        help="Directory where per-case TIR and pass logs are written.",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        default=parse_bool_env("TL_UNALIGNED_REDUCE_ECHO", False),
        help="Print final TIR to stdout in addition to saving files.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available cases and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_cases:
        for idx, case in enumerate(UNALIGNED_REDUCE_CASES):
            print(f"{idx}: {case.label}")
        return

    selected = select_cases(args.case)
    for idx, case in selected:
        print(f"\n--- Lowering case {idx}: {case.label} ---")
        compile_case(idx, case, args)


if __name__ == "__main__":
    main()
