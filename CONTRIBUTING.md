# Contributing to TileLang-Mesh

Bug reports, documentation improvements, tests, and code contributions are welcome. Use the
[SUNMMIO/Tilelang issue tracker](https://github.com/SUNMMIO/Tilelang/issues) for questions and proposed
changes.

## Before Opening a Pull Request

- Search existing issues and pull requests.
- Keep changes focused and include tests for behavior changes.
- Update user documentation when installation, APIs, targets, or behavior change.
- Do not include credentials, private build logs, proprietary hardware details, or generated binaries.

## Clone the Repository

Fork [SUNMMIO/Tilelang](https://github.com/SUNMMIO/Tilelang/fork), then clone your fork:

```bash
git clone git@github.com:<your-user>/Tilelang.git
cd Tilelang
git remote add upstream https://github.com/SUNMMIO/Tilelang.git
```

Contributors without NPU-IR access should initialize only public submodules:

```bash
git submodule update --init --recursive \
  3rdparty/tvm 3rdparty/cutlass 3rdparty/composable_kernel
```

Authorized SUNMMIO contributors can initialize all submodules:

```bash
git submodule update --init --recursive
```

## Development Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel "build[uv]"
python -m pip install -r requirements-dev.txt
pre-commit install --install-hooks
```

Install a public-backend editable build:

```bash
CMAKE_ARGS="-DTILELANG_UPDATE_SUBMODULES=OFF -DUSE_SUNMMIO=OFF" \
  python -m pip install --no-build-isolation --editable . -v
```

Authorized SUNMMIO build:

```bash
CMAKE_ARGS="-DUSE_SUNMMIO=ON" \
  python -m pip install --no-build-isolation --editable . -v
```

The distribution is `tilelang-mesh` and the import package is `tilelang`. Do not install upstream
`tilelang` in the same environment. Uninstall this project with:

```bash
python -m pip uninstall tilelang-mesh
```

## Checks

Run formatting and static checks before submitting:

```bash
pre-commit run --all-files --show-diff-on-failure
```

Select tests appropriate to the changed backend. Public contributors can disable SUNMMIO; tests
marked `sunmmio_closed_runtime` require access-controlled dependencies.

```bash
pytest testing/python/<relevant-test-file>.py
```

For C++ changes, configure and run the repo-local C++ tests when applicable:

```bash
cmake -S . -B build -G Ninja -DTILELANG_BUILD_CPP_TESTS=ON
ninja -C build
ctest --test-dir build --output-on-failure
```

Build documentation with warnings treated as errors:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Packaging changes should build and validate a wheel, then install it from outside the source tree.
Source-distribution validation remains deferred until the NPU-IR packaging boundary is resolved.

## Pull Request Content

Describe:

- What changed and why
- Supported and affected backends
- Tests run and their results
- API or behavior compatibility impact
- Documentation updates
- Any remaining limitation

Release-facing changes should add an entry to [CHANGELOG.md](CHANGELOG.md).
