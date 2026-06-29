<div align="center">

# Tile Language for Distributed Memory
[![PyPI version](https://badge.fury.io/py/tilelang.svg)](https://badge.fury.io/py/tilelang)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tile-ai/tilelang) [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/TUrHyJnKPG)

</div>

## The Goal
The goal of this open source project is to extend Tilelang (https://tilelang.com/) as a unified DSL (domain-specific language) to enable high-performance kernel development for Near-Memory Computing, Distributed Memory AI Accelerators, and Networked Accelerators.

Near-memory computing and distributed memory systems have become key approaches to address the huge computing demand of AI, while networked accelerators further promote the decoupling and coordination of computational resources. To support efficient programming for such emerging heterogeneous computing architectures, we need a unified domain-specific language (DSL) aimed at enabling high-performance kernel development. The design goal is to abstract away underlying hardware differences among different Near-Memory Accelerators or Networked Accelerators, and provide a unified interface, allowing developers to focus on algorithm optimization rather than hardware adaptation. The language likely incorporates key techniques such as tensor tiling, dataflow scheduling, memory layouting, and communication-aware compilation, supporting automatic code generation and optimization to achieve efficient kernel execution across various advanced AI acceleration architectures. Through this unified language framework, we aim to significantly reduce the complexity of cross-platform AI operator development, improving both development efficiency and system performance.

## Latest News
- 10/30/2025 📦:

## HiGHS for ILP Pipeline
The Sunmmio ILP pipeline passes depend on the HiGHS C++ library at build time.

This repository manages HiGHS as a source dependency:

```text
3rdparty/highs/
  ... HiGHS source submodule ...
build/3rdparty/highs-install/
  ... generated install prefix ...
```

Before building, fetch submodules:

```bash
git submodule update --init --recursive
```

During the TileLang build, CMake will automatically build HiGHS from
`3rdparty/highs` and install it to `build/3rdparty/highs-install`.

The generated install prefix under `build/3rdparty/highs-install` is a build
artifact and should not be committed.
