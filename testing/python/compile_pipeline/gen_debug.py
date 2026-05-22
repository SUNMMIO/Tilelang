import argparse
import os
import traceback

from compile_pipeline import compile_test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=str,
        default="all",
        choices=[
            "all",
            "summa",
            "sync",
            "comm",
            "flashattn",
            "mma_3times",
            "overall",
        ],
    )
    parser.add_argument("--target", type=str, default="Sunmmio")
    parser.add_argument("--show-meta", action="store_true")
    parser.add_argument("--remove-header", action="store_false")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--log-passes", type=str, default="")
    args = parser.parse_args()

    base_log_dir = args.log_dir.strip()
    if not base_log_dir:
        base_log_dir = os.path.dirname(__file__)
    elif not os.path.isabs(base_log_dir):
        base_log_dir = os.path.join(os.path.dirname(__file__), base_log_dir)

    log_passes = None
    if args.log_passes.strip():
        log_passes = [p.strip() for p in args.log_passes.split(",") if p.strip()]

    requested_cases = (
        [args.case]
        if args.case != "all"
        else [
            "summa",
            "sync",
            "comm",
            "flashattn",
            "mma_3times",
            "overall",
        ]
    )

    any_failed = False
    for case in requested_cases:
        log_dir = os.path.join(base_log_dir, f"_debug/{case}")
        try:
            if case == "summa":
                from test_summa import summa_matmul

                func = summa_matmul(128, 128, 128, 32, 32, 32)
                out_idx = None
            elif case == "sync":
                from test_sync import kernel_sync

                func = kernel_sync(1024 * 16, 1024 * 16, 1024 * 16, 1024, 1024, 1024)
                out_idx = [2]
            elif case == "sync_2":
                from test_sync_2 import kernel as kernel_sync_2

                func = kernel_sync_2(128, 128, 128, 32, 32, 32)
                out_idx = [2]
            elif case == "comm":
                from test_comm import kernel_comm

                func = kernel_comm(1024 * 16, 1024 * 16, 1024 * 16, 1024, 1024, 1024)
                out_idx = [2]
            elif case == "flashattn":
                from test_flashattn import kernel_flashattn

                func = kernel_flashattn(
                    8,
                    32,
                    4096,
                    128,
                    False,
                    block_M=128,
                    block_N=128,
                    num_stages=1,
                    threads=1,
                )
                out_idx = None
            elif case == "mma_3times":
                from test_mma_3times import kernel_mma_3times_single_thread

                func = kernel_mma_3times_single_thread(1024, 1024, 1024)
                out_idx = None
            elif case == "overall":
                from test_overall import kernel_overall

                func = kernel_overall(128, 128, 128, 64, 64, 32)
                out_idx = [2]
            elif case == "barrier_cf":
                from test_barrier_cf import kernel_barrier_cf

                func = kernel_barrier_cf(128, 128, 32)
                out_idx = [1]
            else:
                raise ValueError(f"Unsupported case: {case}")

            compile_test(
                func,
                out_idx=out_idx,
                target=args.target,
                test_config={},
                log_pass_output=True,
                show_meta=args.show_meta,
                log_dir=log_dir,
                remove_header=args.remove_header,
                log_passes=log_passes,
            )
        except Exception:
            any_failed = True
            traceback.print_exc()

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
