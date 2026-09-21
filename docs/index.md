# TileLang-Mesh Documentation

[GitHub](https://github.com/SUNMMIO/Tilelang) | [Installation](get_started/Installation.md) |
[Changelog](https://github.com/SUNMMIO/Tilelang/blob/tilelang_mesh_main/CHANGELOG.md)

TileLang-Mesh extends TileLang with mesh-aware placement, communication, and compiler support for
distributed-memory and SunMMIO accelerators. The Python distribution is `tilelang-mesh` and the import
package is `tilelang`; it must not be installed alongside the upstream `tilelang` distribution.

:::{toctree}
:maxdepth: 2
:caption: GET STARTED

get_started/Installation
get_started/overview
get_started/targets
:::

:::{toctree}
:maxdepth: 1
:caption: TUTORIALS

tutorials/debug_tools_for_tilelang
tutorials/auto_tuning
tutorials/logging
:::

:::{toctree}
:maxdepth: 1
:caption: PROGRAMMING GUIDES

programming_guides/overview
programming_guides/language_basics
programming_guides/instructions
programming_guides/control_flow
programming_guides/python_compatibility
programming_guides/autotuning
programming_guides/type_system
:::

:::{toctree}
:maxdepth: 1
:caption: SUNMMIO

sunmmio/sunmmio_tilelang_getting_started
sunmmio/sunmmio_tilelang_getting_started_zh_cn
sunmmio/sunmmio_tilelang_user_guide
sunmmio/sunmmio_tilelang_user_guide_zh_cn
sunmmio/pipeline_cost_model_calibration
:::

:::{toctree}
:maxdepth: 1
:caption: DEEP LEARNING OPERATORS

deeplearning_operators/elementwise
deeplearning_operators/gemv
deeplearning_operators/matmul
deeplearning_operators/matmul_sparse
deeplearning_operators/deepseek_mla
:::

:::{toctree}
:maxdepth: 1
:caption: COMPILER INTERNALS

compiler_internals/letstmt_inline
compiler_internals/inject_fence_proxy
compiler_internals/tensor_checks
compiler_internals/sunmmio_tile_loop_fusion
runtime_internals/stubs
:::

:::{toctree}
:maxdepth: 1
:caption: Privacy

privacy
:::
