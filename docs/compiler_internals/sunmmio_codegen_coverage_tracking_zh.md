# SunMMIO Codegen Coverage 标记机制

本文说明 SunMMIO codegen 如何统计 Device TIR 中已经被 lowering 路径处理的
node type 和 call op，以及下面这条 warning 是怎样产生的：

```text
CodeGenTileLangSunMMIO coverage gaps: missing_nodes=N, missing_call_ops=M
```

重点不是列举每一种 TIR node 的 lowering 规则，而是回答以下问题：

- build 从哪里进入 coverage 统计；
- Statement 和 Expression 分别经过什么调用链；
- node type 在哪里被 mark；
- `CallNode` 包装的 op 如何被取出、传递并 mark；
- 为什么 `T.Tiles`、`tir.ret` 和直接消费 AST 的 helper 需要特殊处理；
- warning 中的两个数字具体代表什么。

## 1. 先理解 TIR AST 中的 Node、Stmt 和 Expr

coverage 统计的是 TIR AST 中出现的对象。要理解 node 在哪里被 mark，首先需要区分
AST 节点的类别、节点之间的包含关系，以及 codegen 用来处理节点的函数。

### 1.1 Statement 和 Expression 是并列的节点家族

TIR AST 的主要类型关系可以简化为：

```text
tvm::Object
├── tir::StmtNode                         Statement 节点家族
│   ├── ForNode
│   ├── SeqStmtNode
│   ├── EvaluateNode
│   ├── IfThenElseNode
│   ├── LetStmtNode
│   ├── BufferStoreNode
│   └── ...
│
└── BaseExprNode                         Expression 节点家族
    └── PrimExprNode
        ├── IntImmNode
        ├── VarNode
        ├── AddNode
        ├── MulNode
        ├── CallNode
        ├── BufferLoadNode
        └── ...
```

[`StmtNode`](../../3rdparty/tvm/include/tvm/tir/stmt.h#L37) 是所有 TIR Statement
节点的基类；[`PrimExprNode`](../../3rdparty/tvm/include/tvm/ir/expr.h#L79) 是 TIR
primitive Expression 节点的基类。它们是并列的两个家族，不是谁继承谁：

```text
错误：StmtNode -> PrimExprNode
错误：PrimExprNode -> StmtNode
正确：StmtNode 和 PrimExprNode 都是不同类型的 AST node
```

### 1.2 `Stmt`/`PrimExpr` 是引用，带 `Node` 的类才保存节点数据

TVM 的 C++ 对象模型通常把实际节点和受管理引用分成两层：

| 节点数据类型 | 引用包装类型 | 含义 |
| --- | --- | --- |
| `StmtNode` | `Stmt` | 任意 Statement 节点及其通用引用 |
| `EvaluateNode` | `Evaluate` | 具体 Evaluate 节点及其引用 |
| `PrimExprNode` | `PrimExpr` | 任意 primitive Expression 节点及其通用引用 |
| `CallNode` | `Call` | 具体 Call Expression 节点及其引用 |

例如 [`StmtNode` 和 `Stmt`](../../3rdparty/tvm/include/tvm/tir/stmt.h#L37) 的关系是：

```cpp
class StmtNode : public Object { /* 实际节点数据 */ };
class Stmt : public ObjectRef { /* 指向 StmtNode 的受管理引用 */ };
```

因此函数参数 `const tir::Stmt &stmt` 接收的是引用包装；运行时被它引用的具体对象可能
是 `ForNode`、`EvaluateNode`、`SeqStmtNode` 等。日常说“一个 `Stmt` node”是方便的
简称，更严格的说法是“`Stmt` 引用所指向的具体 `StmtNode` 子类对象”。`PrimExpr` 与
`PrimExprNode` 也是相同关系，定义见
[`ir/expr.h`](../../3rdparty/tvm/include/tvm/ir/expr.h#L120)。

### 1.3 节点之间的 child 关系由字段类型决定

Statement 和 Expression 虽然在继承关系上并列，但它们可以通过字段形成包含关系：

- Statement 可以包含子 Statement；
- Statement 可以包含 Expression；
- Expression 通常继续包含子 Expression；
- 并非节点的每个字段都是 child node，字段也可能是整数、枚举、字符串、dtype、
  annotation 或容器。

常见节点的字段关系如下：

| 节点 | Statement child | Expression child |
| --- | --- | --- |
| `SeqStmtNode` | `seq: Array<Stmt>` | 无 |
| `ForNode` | `body: Stmt` | `loop_var`、`min`、`extent`、`step` |
| `IfThenElseNode` | `then_case`、`else_case` | `condition` |
| `LetStmtNode` | `body` | `value` |
| `EvaluateNode` | 无 | `value` |
| `BufferStoreNode` | 无 | `value`、`indices` |
| `AddNode` | 无 | `a`、`b` |
| `CallNode` | 无 | `args`；另有被调用目标 `op` |

例如 [`ForNode`](../../3rdparty/tvm/include/tvm/tir/stmt.h#L725) 同时具有
`PrimExpr min`、`PrimExpr extent` 和 `Stmt body`。它的 `body` 在运行时可以是任意
具体 Statement，例如 `EvaluateNode`、`SeqStmtNode`、另一个 `ForNode` 或
`IfThenElseNode`。所以不能说“Stmt 的 child 就是 Evaluate”；只能说“某个 `Stmt`
类型的 child 可能指向 `EvaluateNode`”。

### 1.4 `EvaluateNode` 是具体 Statement，`EvalExpr()` 不是 node

这两个名字很相似，但属于不同层次：

- [`EvaluateNode`](../../3rdparty/tvm/include/tvm/tir/stmt.h#L469) 是 TIR AST 中一种具体
  的 `StmtNode`；
- `T.evaluate(expr)` 是 TIR Script 用来构造 `EvaluateNode` 的语法；
- [`EvalExpr()`](../../src/target/sunmmio/codegen_sunmmio.cc#L607) 是 SunMMIO codegen
  用来 lowering `PrimExpr` 的 C++ 方法，不是 AST node；
- TIR 中没有一个通用的 `EvalNode` 类型。

`EvaluateNode` 的核心结构只有一个 Expression 字段：

```cpp
class EvaluateNode : public StmtNode {
 public:
  PrimExpr value;
};
```

它类似 C/C++ 中结果被丢弃的表达式语句：

```cpp
foo();  // 调用表达式作为一条语句
x + 1;  // 表达式结果未被使用
```

TIR 用 `EvaluateNode` 把一个 Expression 放到需要 Statement 的位置，尤其常用于包装
有副作用的 `CallNode`。其构造函数接收一个 `PrimExpr`，见
[`Evaluate::Evaluate`](../../3rdparty/tvm/src/tir/ir/stmt.cc#L428)。如果 `value` 没有
副作用，这条 Evaluate 语句通常可以被优化删除。

### 1.5 用一棵小 AST 串起这些关系

下面的 TIR Script：

```python
for i in T.serial(0, 8):
    T.evaluate(i + 1)
```

可以近似展开成：

```text
PrimFuncNode
└── body: ForNode                         Statement node
    ├── loop_var: VarNode                 Expression node
    ├── min: IntImmNode(0)                Expression node
    ├── extent: IntImmNode(8)             Expression node
    └── body: EvaluateNode                Statement node
        └── value: AddNode                Expression node
            ├── a: VarNode(i)             Expression node
            └── b: IntImmNode(1)          Expression node
```

这棵树同时包含三种常见边：

```text
Statement -> Statement    ForNode.body -> EvaluateNode
Statement -> Expression   EvaluateNode.value -> AddNode
Expression -> Expression  AddNode.a/b -> VarNode/IntImmNode
```

对应到 SunMMIO codegen，调用链是：

```text
VisitStmtTracked(ForNode)
├── mark ForNode
└── VisitStmt_(ForNode)
    ├── EvalExpr(min)
    ├── EvalExpr(extent)
    └── VisitStmtTracked(body = EvaluateNode)
        ├── mark EvaluateNode
        └── VisitStmt_(EvaluateNode)
            └── EvalExpr(value = AddNode)
                ├── MarkVisitedExprRoot(AddNode)
                └── VisitExpr_(AddNode)
                    └── 对两个 child 分别调用 EvalExpr
```

因此 `VisitStmtTracked` 负责进入 Statement 家族，`EvalExpr` 负责进入 Expression
家族；`EvaluateNode` 正好是从 Statement 处理切换到其 `value` Expression 处理的一个
简单例子。它的 handler 最终调用 `EvalExpr(op->value)`，见
[`VisitStmt_(EvaluateNode)`](../../src/target/sunmmio/codegen_sunmmio.cc#L1705)。

### 1.6 coverage 使用的四个集合

`CodeGenTileLangSunMMIO` 同时继承 TVM 的 `StmtVisitor` 和 `ExprFunctor`，见
[`codegen_sunmmio.h`](../../src/target/sunmmio/codegen_sunmmio.h#L310)。它维护四个
字符串集合：

| 集合 | 内容 | 来源 |
| --- | --- | --- |
| `expected_node_types_` | build 前的 Device TIR 中出现过的 node type key | 对 `PrimFunc.body` 做完整后序扫描 |
| `visited_node_types_` | 至少进入过 conversion 路径的 node type key | Statement、Expression 和特殊 helper 的前序 mark |
| `expected_call_ops_` | build 前的 `CallNode` 所引用的 op 名称 | 对 `PrimFunc.body` 做完整后序扫描 |
| `visited_call_ops_` | lowering 路径确认处理过的 call op 名称 | 从已 mark 的 `CallNode::op` 中提取 |

这些容器是 `std::set<std::string>`，声明见
[`codegen_sunmmio.h`](../../src/target/sunmmio/codegen_sunmmio.h#L492)。因此统计的是
**种类**，不是 AST node 实例数。例如 100 个 `tir.Add` 仍然只占一个 node type；
20 次调用 `tl.tileop.copy` 仍然只占一个 call op。

### 1.7 node type 和 call op 不是同一层概念

以下是一个简化的 TIR 片段：

```text
tir.Evaluate                         <- Statement node type
└── tir.Call                         <- Expression node type
    ├── op = tl.tileop.copy          <- call op
    └── args
        ├── tir.Call                 <- Expression node type
        │   └── op = tl.tileop.region <- call op
        └── ir.IntImm                <- Expression node type（以运行时 type key 为准）
```

`tir.Call` 只说明这个 AST 对象是一次调用；`tl.tileop.copy` 才说明它调用了什么。
因此同一个 `tir.Call` node type 可以对应很多不同的 call op。

node type 必须从对象的 [`GetTypeKey()`](../../src/target/sunmmio/codegen_sunmmio.cc#L647)
获取。不能根据 C++ 类名拼接字符串：例如 C++ 类型是 `tir::IntImmNode`，当前运行时
注册的 type key 可能是 `ir.IntImm`，并不保证是 `tir.IntImm`。expected 集合使用的
正是运行时 type key；手写错误前缀只会向 visited 集合插入一个无关字符串，不能抵消
真正缺失的 type key。

## 2. 整体生命周期

一次 `target.build.tilelang_sunmmio_without_compile` 的主要调用链如下：

```text
BuildTileLangSunMMIOWithoutCompile(mod, target, backend)
└── 对 module 中的每个 PrimFunc
    └── CodeGenTileLangSunMMIO::AddFunction(gvar, f)
        ├── CollectExpectedCoverage(f)       # build 前，收集 expected
        ├── 建立函数、参数和 buffer 绑定
        └── VisitStmtTracked(f->body)        # 实际 TIR -> SUVM lowering
└── CodeGenTileLangSunMMIO::Finish()
    ├── WriteCoverageReport()
    ├── CheckCoverageOrFail()
    └── builder_->Finish()
```

入口实现位于
[`BuildTileLangSunMMIOWithoutCompile`](../../src/target/sunmmio/rt_mod_sunmmio.cc#L68)：
它为每个 `PrimFunc` 调用
[`AddFunction`](../../src/target/sunmmio/rt_mod_sunmmio.cc#L82)，最后调用
[`Finish`](../../src/target/sunmmio/rt_mod_sunmmio.cc#L85)。

在 `AddFunction` 内部，expected 扫描发生在
[`CollectExpectedCoverage(f)`](../../src/target/sunmmio/codegen_sunmmio.cc#L792)，
实际函数 body 的 lowering 从
[`VisitStmtTracked(f->body)`](../../src/target/sunmmio/codegen_sunmmio.cc#L832)
开始。两者看到的是同一棵输入 TIR，但使用的遍历机制和目的不同。

如果一个 module 有多个 `PrimFunc`，四个集合会累计这些函数中出现的种类，最后在
`Finish()` 中统一比较。

## 3. expected 是如何收集的

[`CollectExpectedCoverage`](../../src/target/sunmmio/codegen_sunmmio.cc#L676) 对
`f->body` 调用 `tir::PostOrderVisit`：

```text
PostOrderVisit(f->body)
└── 对每个可达 ObjectRef
    ├── obj->GetTypeKey() -> expected_node_types_
    └── 如果 obj 是 CallNode
        ├── OpNode.name -> expected_call_ops_
        ├── GlobalVarNode.name_hint -> "global::<name>"
        └── 其他 target -> "unknown_call_target"
```

对应代码分别在
[`expected_node_types_` 插入处](../../src/target/sunmmio/codegen_sunmmio.cc#L681)
和 [`CallNode::op` 分类处](../../src/target/sunmmio/codegen_sunmmio.cc#L682)。

这次扫描与真正的 codegen 无关。它不会发射 SUVM，也不会调用各个
`VisitStmt_`/`VisitExpr_` lowering handler；它只是给输入 AST 建立一个“出现过哪些
类型和 op”的基线。因此 `CollectExpectedCoverage()` 会看到函数 body 下所有能够由
`PostOrderVisit` 到达的 Statement、Expression 和其他对象。

## 4. Statement 的正常调用链

### 4.1 `VisitStmtTracked` 先 mark 当前 Statement，再做类型分发

Statement 的统一入口是
[`VisitStmtTracked`](../../src/target/sunmmio/codegen_sunmmio.cc#L616)：

```text
VisitStmtTracked(stmt)
├── stmt->GetTypeKey()
│   └── MarkVisitedNodeType(type_key)
│       └── visited_node_types_.insert(type_key)
└── tir::StmtVisitor::VisitStmt(stmt)
    └── StmtFunctor vtable 根据实际 node 类型分发
        └── CodeGenTileLangSunMMIO::VisitStmt_(const XxxNode*)
```

`MarkVisitedNodeType` 本身只做集合插入，见
[`codegen_sunmmio.cc`](../../src/target/sunmmio/codegen_sunmmio.cc#L623)。真正的动态类型
分发由 TVM 的
[`StmtFunctor::VisitStmt`](../../3rdparty/tvm/include/tvm/tir/stmt_functor.h#L80)
完成，其 vtable 注册了 `ForNode`、`EvaluateNode`、`SeqStmtNode` 等类型，见
[`InitVTable`](../../3rdparty/tvm/include/tvm/tir/stmt_functor.h#L107)。

这里最容易产生的误解是：`StmtVisitor::VisitStmt(stmt)` **只负责对当前根节点做类型
分发**，并不会自动替 SunMMIO codegen 递归整棵子树。对子 Statement 和 Expression
的递归由具体 `VisitStmt_` handler 显式发起。

### 4.2 以 `SeqStmt -> Evaluate` 为例

[`VisitStmt_(SeqStmtNode)`](../../src/target/sunmmio/codegen_sunmmio.cc#L900) 对序列中的
每个子 Statement 再调用 `VisitStmtTracked`：

```text
VisitStmtTracked(SeqStmt)
├── mark "tir.SeqStmt"
└── VisitStmt_(SeqStmtNode)
    └── VisitStmtTracked(Evaluate)
        ├── mark "tir.Evaluate"
        └── VisitStmt_(EvaluateNode)
            └── EvalExpr(Evaluate.value)
```

因此 `EvaluateNode` 的 node type 在进入它自己的 handler **之前**，已经由上层的
`VisitStmtTracked` mark。`VisitStmt_(EvaluateNode)` 不需要再次 mark `tir.Evaluate`；
它负责处理 `Evaluate.value` 这个 Expression，普通路径见
[`EvalExpr(op->value)`](../../src/target/sunmmio/codegen_sunmmio.cc#L1718)。

同样的模式也用于普通 `For` 的 body、`IfThenElse` 的两个分支、`While`、`LetStmt`
和 `AttrStmt`。例如普通循环在 `EmitFor` 中调用
[`VisitStmtTracked(op->body)`](../../src/target/sunmmio/codegen_sunmmio.cc#L1531)，
条件分支在 `EmitIf` 中调用
[`VisitStmtTracked(op->then_case)`](../../src/target/sunmmio/codegen_sunmmio.cc#L1556)。

## 5. Expression 的正常调用链

### 5.1 `EvalExpr` 是 Expression 入口，不是 PrimFunc 入口

[`EvalExpr`](../../src/target/sunmmio/codegen_sunmmio.cc#L607) 接收的是
`tvm::PrimExpr`，不是 `PrimFunc`：

```text
EvalExpr(expr)
├── MarkVisitedExprRoot(expr)
│   ├── expr->GetTypeKey() -> visited_node_types_
│   └── 如果根是 CallNode，提取 call->op -> visited_call_ops_
└── ExprFunctor::VisitExpr(expr)
│   └── vtable 根据根节点类型分发
│       └── VisitExpr_(const XxxNode*)
│           ├── 发射当前表达式对应的 SUVM
│           └── 需要时对 child 调用 EvalExpr(child)
```

TVM 的 [`ExprFunctor::VisitExpr`](../../3rdparty/tvm/include/tvm/tir/expr_functor.h#L112)
同样只对当前根节点进行 vtable 分发；支持的节点表在
[`ExprFunctor::InitVTable`](../../3rdparty/tvm/include/tvm/tir/expr_functor.h#L159)。
它不会自动递归 children。

递归发生在 SunMMIO 的具体 handler/helper 中。例如二元运算最终对左右操作数调用
[`EvalExpr(lhs)` 和 `EvalExpr(rhs)`](../../src/target/sunmmio/codegen_sunmmio.cc#L2233)；
`CallNode` 被分发到
[`VisitExpr_(CallNode)`](../../src/target/sunmmio/codegen_sunmmio.cc#L1857)，再进入
[`EmitCall`](../../src/target/sunmmio/codegen_sunmmio.cc#L2697)。

### 5.2 为什么只 mark 当前 Expression 根

[`MarkVisitedExprRoot`](../../src/target/sunmmio/codegen_sunmmio.cc#L638) 只记录传入
`EvalExpr` 的当前根节点；它不会用 `PostOrderVisit` 扫描整棵子树。mark 发生在类型
分发之前，保持与 `VisitStmtTracked` 一致的前序语义。

child 是否被 mark，取决于实际 conversion 是否继续对该 child 调用 `EvalExpr`。例如
二元运算会分别 lowering 两个操作数，所以每个 child 都在自己的 `EvalExpr` 入口被
mark；如果 Analyzer 或 region helper 直接消费并折叠一个输入，则只在 helper 入口
显式 mark 它实际处理的根，不推断其全部后代都被一一转换。

这条规则避免把“存在于输入 AST”误当成“已经单独完成 conversion”。尤其当 helper
把一棵表达式折叠成常量时，内部 child 可能从未对应到任何 SUVM value，不应通过整树
补扫被标记。

coverage 容器仍然是 type/op 集合，而不是实例集合。因此它回答的是“这种 type/op
是否至少进入过一条 conversion 路径”，不能证明同类型的每个 AST 实例都被处理。

## 6. `CallNode` 包装的 op 如何被传递并 mark

### 6.1 `CallNode` 同时携带“调用节点”和“被调用目标”

概念上可以把它看成：

```cpp
CallNode {
  ObjectRef op;         // Op 或 GlobalVar
  Array<PrimExpr> args;
  DataType dtype;
}
```

`CallNode` 自己贡献一个 node type，例如 `tir.Call`；`call->op` 指向的对象贡献一个
call op，例如 `tir.if_then_else`、`tl.tileop.copy` 或某个 `GlobalVar`。

### 6.2 从当前 Call 根到 op 字符串的完整链条

正常 Expression 路径中的传递过程如下：

```text
EvalExpr(original_expr)
└── MarkVisitedExprRoot(original_expr)
    ├── MarkVisitedNodeType(original_expr->GetTypeKey())
    └── MarkVisitedCallOpFromExpr(original_expr)
        ├── original_expr.as<tir::CallNode>()
        └── 如果根是 Call，检查 call->op
            ├── OpNode      -> op_node->name
            ├── GlobalVar   -> "global::" + name_hint
            └── 其他        -> "unknown_call_target"
                └── visited_call_ops_.insert(...)
```

`MarkVisitedExprRoot` 把当前 `PrimExpr` 直接传给
[`MarkVisitedCallOpFromExpr`](../../src/target/sunmmio/codegen_sunmmio.cc#L624)。后者用
`expr.as<tir::CallNode>()` 判断当前根是不是 Call，并从
[`call->op`](../../src/target/sunmmio/codegen_sunmmio.cc#L633) 中提取稳定的名称。
因此“op 被传到 mark 函数”并不是把字符串沿着所有 visitor 层层传参，而是在已经找到
`CallNode` 后，从该 node 自己保存的 `op` 字段现场取出。

### 6.3 lowering 的 `EmitCall` 与 coverage mark 是两条相关但不同的逻辑

[`EmitCall`](../../src/target/sunmmio/codegen_sunmmio.cc#L2697) 也会读取同一个
`op->op` 字段，将它转换成局部变量 `callee`，然后根据名称选择具体 SUVM lowering
分支，见 [`callee` 提取](../../src/target/sunmmio/codegen_sunmmio.cc#L2702)。

但 `EmitCall` 的分支选择本身不等于 coverage mark。普通路径在进入 `EmitCall` 前，
外层 `EvalExpr` 已通过 `MarkVisitedExprRoot` 记录当前 Call 和它包装的 op；嵌套 Call
只有在实际递归调用 `EvalExpr`，或被某个 helper 直接识别时，才会被单独 mark。

## 7. 不走完整通用递归的特殊路径

### 7.1 `T.Tiles` 注解循环

`ForNode` 的通用入口仍然是：

```text
VisitStmtTracked(For)
├── mark ForNode type
└── VisitStmt_(ForNode)
    ├── TryLowerTilesScope(op) == true
    │   └── return，不进入普通 EmitFor
    └── 否则 EmitFor(op)
```

分支代码见
[`VisitStmt_(ForNode)`](../../src/target/sunmmio/codegen_sunmmio.cc#L1564)。因此外层
`ForNode` 在进入 `TryLowerTilesScope` 前已经被 mark，但 Tiles body 不再通过普通
`EmitFor -> VisitStmtTracked(body)` 路径处理。

Tiles lowering 有自己的一套递归器：入口是
[`TryLowerTilesScope`](../../src/target/sunmmio/sunmmio_codegen_tiles_loop.cc#L649)，
内部定义了 [`lower_expr`](../../src/target/sunmmio/sunmmio_codegen_tiles_loop.cc#L2868)、
[`lower_stmt`](../../src/target/sunmmio/sunmmio_codegen_tiles_loop.cc#L3608)、
[`lower_reduce_stmt`](../../src/target/sunmmio/sunmmio_codegen_tiles_loop.cc#L4081)
和 [`emit_loop_nest`](../../src/target/sunmmio/sunmmio_codegen_tiles_loop.cc#L4457)。这些
函数直接分析 TIR 并发射 tile-level SUVM，不保证每个内部 node 都进入通用
`VisitStmtTracked`/`EvalExpr`。

因此 Tiles 自定义递归器在各自的前序入口复用相同的根节点 mark 规则：

- `lower_stmt` / `lower_reduce_stmt` 在分支判断前 mark 当前 Statement type；
- `lower_expr` 在分支判断前调用 `MarkVisitedExprRoot(expr)`；
- 自定义循环在消费 `min`、`extent`、`step` 前 mark 对应 Expression 根；
- region、reduce 等不经过 `lower_expr` 的直接 helper 输入，在 helper 调用点显式 mark；
- 纯 token placeholder 如果被 Tiles lowering 明确忽略，则在 mark 之前返回。

调用链是：

```text
VisitStmtTracked(Tiles For)          # 先 mark 外层 For
└── VisitStmt_(ForNode)
    └── TryLowerTilesScope
        ├── lower_stmt(stmt)
        │   ├── mark 当前 stmt type
        │   └── 只对实际 lowering 的 child 继续递归
        ├── lower_expr(expr)
        │   ├── mark 当前 expr type/call op
        │   └── 只对实际 lowering 的 child 继续递归
        ├── emit_loop_nest
        │   └── 在消费 domain loop 及其边界时分别 mark
        └── return true
```

这里没有“Tiles 完成后补扫整棵原始 `For` 子树”的步骤。这样不会把被重写、折叠或
忽略、没有实际进入 conversion 的 child 误记为 visited。

### 7.2 `Evaluate(tir.ret(0))` 提前返回

普通 `Evaluate` 会调用 `EvalExpr(op->value)`，但 `tir.ret(0)` 是特殊控制语义：
builder 会在函数结尾统一发射 return，因此
[`VisitStmt_(EvaluateNode)`](../../src/target/sunmmio/codegen_sunmmio.cc#L1705) 校验它后
直接返回，不再调用 `EvalExpr`。

```text
VisitStmtTracked(Evaluate)
├── mark EvaluateNode
└── VisitStmt_(EvaluateNode)
    └── value 是 tir.ret
        ├── mark CallNode type
        ├── mark call op "tir.ret"
        ├── 校验唯一参数是 IntImm(0)
        ├── 用 imm->GetTypeKey() mark IntImm
        └── return
```

因为这里绕过了普通 `EvalExpr`，所以必须显式 mark Call、`tir.ret` 和常量参数。Call
根通过 `MarkVisitedExprRoot(op->value)` 记录；常量使用实际对象的 `GetTypeKey()`。对应实现见
[`codegen_sunmmio.cc`](../../src/target/sunmmio/codegen_sunmmio.cc#L1707)。

### 7.3 直接消费参数的 helper

部分 call 参数不是通过 `EvalExpr` 转成 SSA value，而是被 helper 解析成 region、属性
或常量：

- [`TryConsumeSyncTokenId`](../../src/target/sunmmio/codegen_sunmmio.cc#L654) 直接识别
  `tl.sync_token_id(IntImm)`，并显式 mark Call、call op 和实际 IntImm type key；
- [`EmitLocalVarLoad` / `EmitLocalVarStore`](../../src/target/sunmmio/codegen_sunmmio.cc#L1164)
  在 Analyzer 简化索引之前 mark 原始索引根；
- [`EmitRegionCall`](../../src/target/sunmmio/codegen_sunmmio.cc#L2661) 先记录 region call
  根、call op、一级参数和归一化后的 `min/extent` 根，再对需要生成 SSA 的 min 调用
  `EvalExpr`；
- [`EmitMXPackOrUnpack`](../../src/target/sunmmio/codegen_sunmmio.cc#L2310) 在归一化
  region 时 mark region Call、一级参数以及实际检查的 `min/extent` 根。

这些局部 mark 可能与普通 `EvalExpr` 根标记重叠；集合会自动消除重复项。它们不会递归
扫描 helper 输入的全部后代。

新增特殊 lowering 时，应优先遵循下面的选择：

1. 普通 child Expression 需要产生值时，调用 `EvalExpr(child)`；
2. helper 直接消费一个 Expression、但不通过 `EvalExpr` 时，在消费前调用
   `MarkVisitedExprRoot(original_expr)`；
3. helper 直接检查的常量等节点使用对象自身的 `GetTypeKey()`；Call 根统一使用
   `MarkVisitedExprRoot`，同时记录 node type 和 call op；
4. 不要使用猜测或硬编码的 type key；
5. 特殊 Statement 递归完全绕过通用 visitor 时，在每个自定义递归入口前序 mark 当前
   Statement，而不是在结束后 mark 整棵原始子树。

## 8. warning 如何计算

[`CheckCoverageOrFail`](../../src/target/sunmmio/codegen_sunmmio.cc#L751) 计算两个集合差：

```text
missing_node_types = expected_node_types - visited_node_types
missing_call_ops   = expected_call_ops  - visited_call_ops
```

具体的 `std::set_difference` 和赋值见
[`codegen_sunmmio.cc`](../../src/target/sunmmio/codegen_sunmmio.cc#L759)。只要任一差集非空，
就输出 warning：

```text
CodeGenTileLangSunMMIO coverage gaps: missing_nodes=<type 种类数>,
                                      missing_call_ops=<op 种类数>
```

所以：

- `missing_nodes=2` 表示有两种 expected type key 没进入 visited 集合，不是漏了两个
  node 实例；
- `missing_call_ops=1` 表示有一种被 `CallNode` 包装的 op 名称未进入 visited 集合；
- `missing_call_ops=0` 只说明 call op 名称集合完整，不代表 node type 也完整；
- 两类 missing 都可能由同一种根因造成：某条成功的特殊 conversion 路径绕过了通用
  mark 机制。

默认模式只 warning，不会阻止 codegen，因此端到端测试可能仍然通过。设置
`TL_SUNMMIO_CODEGEN_COVERAGE_STRICT=1` 后，同样的缺口会在
[`LOG(FATAL)`](../../src/target/sunmmio/codegen_sunmmio.cc#L771) 处使 build 失败。

如需查看具体缺失项，而不只是两个数字，可设置：

```bash
export TL_SUNMMIO_CODEGEN_COVERAGE_PATH=/tmp/sunmmio-coverage.json
export TL_SUNMMIO_CODEGEN_COVERAGE_STRICT=1
```

[`WriteCoverageReport`](../../src/target/sunmmio/codegen_sunmmio.cc#L700) 会输出 expected、
visited 和 missing 六个数组；环境变量读取位置见
[`TL_SUNMMIO_CODEGEN_COVERAGE_PATH`](../../src/target/sunmmio/codegen_sunmmio.cc#L701)。

## 9. 调试 missing 项的顺序

遇到 coverage warning 时，可以按以下顺序定位：

1. 设置 `TL_SUNMMIO_CODEGEN_COVERAGE_PATH`，先得到具体 type/op 名称；
2. 在原始 Device TIR 中找到该节点，确认它属于 Statement 还是 Expression；
3. 对 Statement 向上追踪谁应调用 `VisitStmtTracked`；
4. 对 Expression 向上追踪谁应调用 `EvalExpr`，或哪个 helper 直接消费了它；
5. 对 missing call op 找到包裹它的 `CallNode`，检查父 Expression 是否最终进入
   `EvalExpr`，或特殊路径是否调用 `MarkVisitedExprRoot`；
6. 如果位于 `T.Tiles` 子树，检查对应节点是否进入 `lower_stmt`、`lower_reduce_stmt`、
   `lower_expr`，或某个显式标记直接输入的 helper；
7. 只在 lowering 确实接受了该 AST 后补 mark，不要在 expected 扫描阶段直接复制到
   visited 集合，否则 coverage 将失去检测 conversion 路径遗漏的意义。

还要注意：这里追踪的是**输入 TIR AST**。`SunMMIOBuilder` 新创建的 SUVM/MLIR op
不属于 expected 或 visited；MLIR builder 发射了某个 op，不能自动证明对应 TIR
node/call op 已被 coverage mark。

## 10. 测试覆盖

coverage 回归测试集中在
[`test_skeleton.py`](../../testing/python/sunmmio/codegen/test_skeleton.py#L639)：测试设置
report 路径和 strict 模式，并要求两个 missing 数组都为空。

三类特殊路径有独立回归测试：

- [helper 直接消费 Expression 根](../../testing/python/sunmmio/codegen/test_skeleton.py#L660)，
  验证被 `EmitLocalVarLoad` 直接简化、没有进入 `EvalExpr` 的 `tir.Mul` 根被记录；
- [`Evaluate(tir.ret(0))` 提前返回](../../testing/python/sunmmio/codegen/test_skeleton.py#L673)，
  验证 `tir.Call` node type 和 `tir.ret` call op 都被记录；
- [真实 `T.Tiles` softmax](../../testing/python/sunmmio/codegen/test_hybrid_loop_opt_validate.py#L99)，
  在 strict 模式下验证自定义 statement/expression 递归和 reduce helper 都没有 missing。

## 11. 快速对照表

| AST 路径 | lowering 入口 | node type 在哪里 mark | call op 在哪里 mark |
| --- | --- | --- | --- |
| 普通 Statement 根 | `VisitStmtTracked` | 分发前 `stmt->GetTypeKey()` | 不适用；Statement 自身不是 Call |
| 普通 Expression 根 | `EvalExpr` | 分发前 `MarkVisitedExprRoot` | 当前根为 Call 时读取 `CallNode::op` |
| 普通 Expression child | 父 handler 显式调用 `EvalExpr(child)` | child 自己的前序入口 | child 根为 Call 时提取 |
| 普通 `CallNode` | `VisitExpr_(CallNode) -> EmitCall` | 进入 `VisitExpr_` 前由 `EvalExpr` mark | 同一次根节点 mark 读取 `CallNode::op` |
| `T.Tiles` For 外层 | `VisitStmtTracked -> VisitStmt_(ForNode)` | `VisitStmtTracked` | 不适用 |
| `T.Tiles` 内部节点 | `lower_stmt` / `lower_reduce_stmt` / `lower_expr` | 各自递归入口前序 mark 当前根 | `lower_expr` 根为 Call 时提取；直接 helper 显式提取 |
| `Evaluate(tir.ret(0))` | `VisitStmt_(EvaluateNode)` 特殊分支 | 显式 mark Call 和 IntImm；Evaluate 已由上层 mark | 显式 `MarkVisitedCallOpFromExpr` |
| sync token helper | `TryConsumeSyncTokenId` | 显式使用实际对象的 `GetTypeKey()` | 显式 `MarkVisitedCallOpFromExpr` |
| local.var/region/MX helper | 对应 helper | 只 mark helper 实际消费的 Expression 根或直接常量 | 直接根为 Call 时提取 `CallNode::op` |

最核心的判断是：**dispatcher 只负责把当前根节点送到正确 handler；递归由 handler
显式组织；coverage mark 又是叠加在这些 lowering 路径上的独立记账机制。** 三者不能
视为同一个动作。
