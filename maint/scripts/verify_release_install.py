#!/usr/bin/env python3
"""Verify an installed TileLang-Mesh distribution outside the source tree."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as distribution_version
import subprocess


PROJECT_DISTRIBUTION = "tilelang-mesh"
UPSTREAM_DISTRIBUTION = "tilelang"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected_version")
    parser.add_argument(
        "--require-sunmmio",
        action="store_true",
        help="verify the SunMMIO code generator and packaged NPU-IR tools",
    )
    args = parser.parse_args()

    installed_version = distribution_version(PROJECT_DISTRIBUTION)
    if installed_version != args.expected_version:
        raise SystemExit(
            f"{PROJECT_DISTRIBUTION} metadata version {installed_version!r} does not match expected version {args.expected_version!r}"
        )

    try:
        upstream_version = distribution_version(UPSTREAM_DISTRIBUTION)
    except PackageNotFoundError:
        pass
    else:
        raise SystemExit(
            f"The upstream {UPSTREAM_DISTRIBUTION!r} distribution is installed alongside "
            f"{PROJECT_DISTRIBUTION} ({upstream_version}); these distributions cannot coexist safely."
        )

    import tilelang

    if tilelang.__version__ != installed_version:
        raise SystemExit(
            f"tilelang.__version__ {tilelang.__version__!r} does not match {PROJECT_DISTRIBUTION} metadata {installed_version!r}"
        )

    if args.require_sunmmio:
        from tilelang import tvm
        from tilelang.jit.adapter.sunmmio.libgen import find_npuir_tool
        from tilelang.utils.target import determine_target

        builder = tvm.ffi.get_global_func(
            "target.build.tilelang_sunmmio_without_compile",
            allow_missing=True,
        )
        if builder is None:
            raise SystemExit("The installed wheel does not provide the SunMMIO SUVM code generator.")

        stmt = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
        func = tvm.tir.PrimFunc([], stmt)
        func = func.with_attr("global_symbol", "main").with_attr(
            "calling_conv",
            int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH),
        )
        target = determine_target("Sunmmio", return_object=True)
        runtime_module = builder(tvm.IRModule({"main": func}), target, "suvm")
        if not runtime_module.inspect_source().strip():
            raise SystemExit("The installed SunMMIO code generator produced empty SUVM output.")

        for tool_name in ("npuir-opt", "npuir-translate", "npuir-compile"):
            tool = find_npuir_tool(tool_name)
            subprocess.run(
                [tool, "--version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

    backend = " with SunMMIO" if args.require_sunmmio else ""
    print(f"Verified {PROJECT_DISTRIBUTION} {installed_version}{backend}")


if __name__ == "__main__":
    main()
