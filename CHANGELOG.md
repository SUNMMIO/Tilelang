# Changelog

All notable user-facing changes to TileLang-Mesh are documented in this file.

The project follows [Semantic Versioning](https://semver.org/) for public releases. Because this is
beta software, minor releases may still contain explicitly documented compatibility changes.

## [0.1.0] - Unreleased

### Highlights

- Initial public beta release of the `tilelang-mesh` Python distribution.
- Distributed-memory and mesh-aware TileLang extensions for placement, communication, layout, and
  code generation.
- SUNMMIO/SUVM backend integration for authorized source builds with NPU-IR access.
- English and Chinese SunMMIO kernel quick-start and user guides.

### Distribution

- Public GitHub Release wheels are built with `USE_SUNMMIO=ON`, package the required NPU-IR
  executables, and do not contain the access-controlled NPU-IR source tree.
- Python source distributions are deferred until the NPU-IR packaging and redistribution boundary
  has been resolved.
- CPython 3.12 or newer is required.
- Legacy CUDA and ROCm container definitions are not included; release artifacts target the
  SunMMIO/SUVM backend.
- Authorized users can enable SUNMMIO from a recursive source checkout.
- The distribution name is `tilelang-mesh`, while the import package remains `tilelang`.

### Known Limitations

- `tilelang-mesh` cannot safely coexist with the upstream `tilelang` distribution in one Python
  environment.
- GitHub-generated "Source code" archives omit required submodule content; use an attached release
  wheel or clone the repository.
- SUNMMIO builds require access to `SUNMMIO/NPU-IR` and may require an LLVM/MLIR source checkout.
- Exact supported Python, operating-system, architecture, and accelerator combinations are listed in
  the GitHub Release notes after artifact verification.
- The previously generated API reference is temporarily omitted because inherited docstrings do not
  yet pass a strict Sphinx build; the maintained guides remain available.

### Release Process

- The `v0.1.0` Tag, `VERSION`, distribution metadata, and `tilelang.__version__` must all resolve to
  `0.1.0` before this section receives a release date.
- Replace `Unreleased` with the publication date only after final wheel smoke tests pass.

[0.1.0]: https://github.com/SUNMMIO/Tilelang/releases/tag/v0.1.0
