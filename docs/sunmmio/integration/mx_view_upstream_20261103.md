# MX view 与 dev/main 整合记录

状态：整合和验证完成。基线为本地 `6419590b` 与上游 `7b95b7cc`，共同基线 `b4bdcb4f`。整合分支 `integration/mx-view-upstream-20261103`。

## 冲突取舍

| 文件 | 双方改动与处理 |
| --- | --- |
| `src/layout/cute_layout.cc` | 本地按物理覆盖推导 MX 字段/padding，并支持高维 slot 与二维矩阵互拷；上游统一布局命名、扩展 batch 布局。保留 covered shape 与降维语义，升维按上游方式平移布局轴，统一为 MXNNZ 名称。 |
| `src/op/copy.cc` | 保留本地 MX shape、整块/整行及起点对齐检查，接入上游非连续 RSRAM ODMA1 计划与 rank-3 bank copy。MX 容器继续使用已有 DMA 路径，不能落入仅支持普通 dtype 的新规划器。布局匹配保留 MX 专用推导；普通 dtype 同时接受上游 singleton-mode 等价判断。 |
| `src/target/sunmmio/codegen_sunmmio.cc` | 合并本地 copy_operand 对齐诊断和上游 preserve_region_rank、allow_rank3、planned tile view。同步检查 header/builder/call 层及所有调用点，显式补出不适用 copy 诊断的参数，避免 bool 实参错位。 |
| `test_copy_region_validate.py` | 合并导入；保留本地 MX 跨维/地址回归和上游 RSRAM strided-copy 测试。 |
| `test_layout_transform_opt_validate.py` | 保留本地 BF16/MXFP8/MXFP4 参数矩阵，同时加入上游非对齐 partial rank-3 测试。 |
| `test_mx_gemm.py` | 保留本地 weight staging、通信和 slot 回归，同时加入上游 greedy/ILP dtype-pair 测试及非法组合诊断。 |
| `test_layout_inference.py` | 合并双方导入与测试，普通 ZN 名称改为上游 NZ，保留本地 MX weight 布局传播检查。 |
| `3rdparty/NPU-IR` | 指向上游 `e7e49427`；旧 pin `2354427f` 是其祖先，没有本地 NPU-IR 源码改动需要合并。临时 worktree 未检出子模块造成的 Git 冲突按该版本关系解决。 |

## 自动合并后的兼容点

- 本地新增的 `sunmmio_layout_inference.cc`、`op/utils.cc` 仍引用旧的 `kMXZNN`/`MakeMXZNN`，同步为 `kMXNNZ`/`MakeMXNNZ`。
- MX 跨 rank 的 slot copy 继续要求前导轴 extent=1；双方都是 rank-3 且有效 shape 匹配时保留上游批量 copy 支持，bank 布局不匹配时沿用上游逐 batch 展开。
- 上游 RSRAM dispatch 的新拒绝分支会挡住本地已有 MX 容器 DMA；显式保留该已支持路径，不对普通 dtype 放宽准入。
- pass 顺序保留 MX binding 规范化和布局解析前后校验，同时接入上游 frontend syntax、batch GEMM view、dynamic put 校验。
- 新增动态 put 校验曾把 MX 字段 handle 替换为 Call，令量化 kernel 的 Buffer backing Var 非法；只将标量 Let 纳入路由证明替换，存储 handle 保持 Var。该适配同时通过 MX quant/dequant 与原 dynamic put 回归。
- `test_cute_layout.py` 的本地新增 factory 参数仍使用旧变量名，同步为 `make_mxnnz`，保留其测试内容与旧 API 别名专用测试。
- 3 个 BF16 非法 copy 在上游规划器中更早被拒绝，报错断言改为该前置拒绝消息；仍要求相同非法输入抛出 InternalError，MX dtype 的 shape/起点对齐断言保持不变。
- 保留 MX view 对相关自动 pipeline 的拒绝规则；上游 MX dtype-pair 支持适用于已有普通 MX pipeline，未据此开放 MX view 多版本存储。

## 验证范围

1. 独立 Release 构建 TileLang、npuir-opt、npuir-compile，LLVM SDK 按固定 revision 核对。
2. 冲突关联测试及 MX view、pack/unpack、跨维 copy、布局传播、GEMM、ODMA1、dynamic put、pipeline 回归。
3. 完整 SunMMIO 编译回归及必要的通用负索引/Let 测试。
4. ELF 链接和 Gem5：MX 量化、字段/padding、父子可见性、上下文索引以及上游新增路径。设备工具链和模拟器版本单独记录。

## 最终验证结果

| 检查 | 结果 |
| --- | --- |
| ✓ PASSED：独立 Release 联合构建 | TileLang、npuir-opt、npuir-compile 成功；TVM pin 保持 `930c59f6`，NPU-IR 为 `e7e49427` |
| ✓ PASSED：完整选定回归 | language/layout/transform/codegen/JIT、batch GEMM/all_reduce/reduce/transpose、四组 pipeline、通用负索引及 LetInline：2162 通过、6 跳过、1 既有 xfailed |
| ✓ PASSED：补测 6 项 JIT | 上一行因缺少设备工具链而跳过的 6 项，在隔离环境全部通过；包含三个 elementwise 执行样例、LLVM/ELF 生成及缓存检查 |
| ✓ PASSED：量化与 MX GEMM 数值 | 8 项，copy/view/CPU 量化反量化对照，以及两种 dtype 的双 MX GEMM |
| ✓ PASSED：row-major 字段及 padding 数值 | 24 项；两种 dtype、pack/unpack/roundtrip、K=32/96/1056/4096，逐核位模式检查 |
| ✓ PASSED：上游新增路径数值 | 5 项：GEMM v1/v2 的 rank-3 批量输出与 batch reduction，以及 strided ODMA1；对照 CPU 并检查 16 核 |
| ✓ PASSED：父子可见性与上下文索引 | 14 项：两种 dtype × 三种布局的父存储两次更新，以及字段 Let/循环索引的双向 copy，逐核检查 |
| ✓ PASSED：格式和静态检查 | 49 个变更 Python 文件通过 Ruff check/format；手工整合 C++ 修改行通过 clang-format；无冲突标记、无 diff 空白错误 |
| ○ SKIPPED：未执行范围 | NPU-IR 独立全量 lit/GTest、Debug 构建和性能测量；底层源码未修改，本次验证其 TileLang 调用链与相关设备数值 |

数值测试合计 51 项全部通过，均使用最终整合库。6 项 JIT 补测单列，不重复计入这 51 项。唯一 xfail 是已有的通用 codegen 测试绕过生产 lowering 管线的情形，未改变其标记。

初次全量回归中发现并处理：动态 put 对字段 handle 的错误替换、布局测试旧变量名、3 个 BF16 非法 copy 的前置诊断差异。测试进程的 editable 安装曾将子进程导入指向原分支，已用隔离启动配置纠正并重跑；最终结果核对了实际 Python、库和驱动路径，没有使用旧分支产物冒充整合产物。

## 设备环境与依赖

- 使用独立的 Sunmmio toolchain **v0.2.9**，clang 提交 `c71533896147a2ee9af3df3542d13c9575f2421a`，以及独立 Python 目录中的 **sunsim 0.0.7**。
- Gem5 沿用可用的 `0.260716+7115.857f9e8c.py310`；本次所有设备数值及 JIT 补测均使用上述实际组合，没有修改共享服务或原工具链安装。
- 旧 sunsim 0.0.2 要求 `.kernargs` 段，与上游的新链接约定不兼容。保留上游 ABI，更新隔离验证环境；没有通过更改编译器产物绕过失败。
- 外部数值脚本的 `Layout.zz` 调整为 sunsim 0.0.7 的 `layout.blockwise`，保持相同布局、输入、参考和逐位断言。
- LLVM SDK 的 revision 与 NPU-IR pin 精确一致：`c495a9b796c46456e523a5eeb3690415ba99f5db`。底层 preflight 脚本不能解析 SDK 头文件中的原始字符串，报告未验证；已直接核对完整 revision 并完成联合构建，未将脚本记为通过。
- 新 NPU-IR 的 `tile.quant/tile.dequant` canonicalization 与当前无 mask 量化 cast 路径已通过上述编译及数值验证。

最终 `libtilelang.so` SHA256：`577427cf06caab126773a4c4cea1f02b75fdfb5bf703357eb1332ed3e30f1818`。
最终 `npuir-compile` SHA256：`ba366d2a04c59895e41e0a1d9e2447a86bbb6a5d8031761039724143b445f3db`。

## 交付与证据

全部冲突已解决，没有未决的功能语义选择。整合工作保存在独立分支和 worktree，原分支及其 build 保持可用；未推送远端。

本次证据索引为整合工作区旁的 `integration-final-regression.log/xml`、`integration-fixes.log/xml`、`loaded-compiler.json` 和联合构建日志；设备验证目录包含 `numeric-v029.xml`、`row-major-v029.xml`、`upstream-v029.xml`、`visibility-context-v029.xml`、`jit-v029.xml`，以及对应逐核输出、编译产物、源码/二进制指纹和 `runtime-environment.json`。具体本地目录由交付消息给出。

## 提交前补充修复

以下为合并提交 `9be8bf7a` 之后的局部修复，前文整合及数值结果对应该合并提交。

- DMA 对齐证明在 codegen 入口遍历最终 TIR，复用已有上下文分析，覆盖分支、循环、Let 和 assume；删除 region 发射阶段无上下文的重复检查及其多余参数。实际地址计算保持原样，DMA 和 layout-transform 均沿用原有对齐要求。
- 可变内存读取不作为贯穿作用域的证明，防止索引被改写后误放行；分支约束在离开分支时撤销。
- 含 `T.mx_view` 的函数提前拒绝 `T.annotate_safe_value`，包括普通 peer 上的注解；普通无 view 函数继续支持。错误消息中的 WSRAM 布局名同步为 `MXNNZ`。
- 本轮构建通过；13 项定向用例与 119 项直接相关 copy/layout-transform/MX 应用编译回归全部通过。正向用例比较等价显式对齐写法的完整 SUVM，并通过 LLVM lowering；反向用例验证未对齐、约束泄漏和可变索引仍被拒绝。未重复全量回归或 Gem5。
