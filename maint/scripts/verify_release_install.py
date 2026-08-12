#!/usr/bin/env python3
"""Verify an installed TileLang-Mesh distribution outside the source tree."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as distribution_version


PROJECT_DISTRIBUTION = "tilelang-mesh"
UPSTREAM_DISTRIBUTION = "tilelang"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected_version")
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

    print(f"Verified {PROJECT_DISTRIBUTION} {installed_version}")


if __name__ == "__main__":
    main()
