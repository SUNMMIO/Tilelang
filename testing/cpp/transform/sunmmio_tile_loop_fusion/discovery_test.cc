#include <gtest/gtest.h>

#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include "transform/common/attr.h"
#include "transform/sunmmio_tile_loop_fusion/discovery.h"
#include "transform/sunmmio_tile_loop_fusion/utils.h"

#include <string>
#include <vector>

namespace tvm {
namespace tl {
namespace {

using namespace tir;

PrimExpr I(int value) { return IntImm(DataType::Int(32), value); }

Buffer MakeSharedBuffer(const std::string &name, const Array<PrimExpr> &shape) {
  return decl_buffer(shape, DataType::Float(16), name, "shared.rsram");
}

Map<String, ObjectRef>
MakeScopeEntryAnnotations(std::initializer_list<int> execution_domain_axes) {
  Map<String, ObjectRef> annotations;
  Array<PrimExpr> domain_axes;
  for (int axis : execution_domain_axes) {
    domain_axes.push_back(I(axis));
  }
  annotations.Set(attr::tile_scope_entry, Integer(1));
  annotations.Set(attr::tile_execution_axis, Integer(0));
  annotations.Set(attr::tile_execution_domain_axes, domain_axes);
  annotations.Set(attr::kTileDomain, Array<PrimExpr>{I(32), I(32)});
  annotations.Set(attr::tile_tile_size, Array<PrimExpr>{I(8), I(32)});
  return annotations;
}

Map<String, ObjectRef> MakeExecutionAxisAnnotation(int axis) {
  Map<String, ObjectRef> annotations;
  annotations.Set(attr::tile_execution_axis, Integer(axis));
  return annotations;
}

Map<String, ObjectRef> MakeInteriorAnnotation(int axis) {
  Map<String, ObjectRef> annotations;
  annotations.Set(attr::tile_interior, Integer(1));
  annotations.Set(attr::tile_interior_axis, Integer(axis));
  return annotations;
}

Stmt Make2DTileCopy(const Buffer &dst, const Buffer &src,
                    const std::string &axis0_name = "i",
                    const std::string &axis1_name = "j",
                    std::initializer_list<int> execution_domain_axes = {0, 1}) {
  Var axis0(axis0_name, DataType::Int(32));
  Var axis1(axis1_name, DataType::Int(32));
  Var ki("ki", DataType::Int(32));
  Var kj("kj", DataType::Int(32));

  PrimExpr row = axis0 * I(8) + ki;
  PrimExpr col = axis1 * I(32) + kj;

  Stmt body = BufferStore(dst, BufferLoad(src, {row, col}), {row, col});
  body = For(kj, 0, I(32), ForKind::kVectorized, body, Optional<IterVar>(),
             MakeInteriorAnnotation(1));
  body = For(ki, 0, I(8), ForKind::kSerial, body, Optional<IterVar>(),
             MakeInteriorAnnotation(0));
  body = For(axis1, 0, I(1), ForKind::kSerial, body, Optional<IterVar>(),
             MakeExecutionAxisAnnotation(1));
  body = For(axis0, 0, I(4), ForKind::kSerial, body, Optional<IterVar>(),
             MakeScopeEntryAnnotations(execution_domain_axes));
  return body;
}

Map<String, ObjectRef> MakeScopeEntryAnnotationsWithTileSize(
    std::initializer_list<int> execution_domain_axes, int tile_rows,
    int tile_cols, int domain_rows = 32, int domain_cols = 32) {
  Map<String, ObjectRef> annotations;
  Array<PrimExpr> domain_axes;
  for (int axis : execution_domain_axes) {
    domain_axes.push_back(I(axis));
  }
  annotations.Set(attr::tile_scope_entry, Integer(1));
  annotations.Set(attr::tile_execution_axis, Integer(0));
  annotations.Set(attr::tile_execution_domain_axes, domain_axes);
  annotations.Set(attr::kTileDomain,
                  Array<PrimExpr>{I(domain_rows), I(domain_cols)});
  annotations.Set(attr::tile_tile_size,
                  Array<PrimExpr>{I(tile_rows), I(tile_cols)});
  return annotations;
}

Stmt Make2DTileRead(const Buffer &src, const std::string &axis0_name = "i",
                    const std::string &axis1_name = "j",
                    std::initializer_list<int> execution_domain_axes = {0, 1}) {
  Var axis0(axis0_name, DataType::Int(32));
  Var axis1(axis1_name, DataType::Int(32));
  Var ki("ki", DataType::Int(32));
  Var kj("kj", DataType::Int(32));

  PrimExpr row = axis0 * I(8) + ki;
  PrimExpr col = axis1 * I(32) + kj;

  Stmt body = Evaluate(BufferLoad(src, {row, col}));
  body = For(kj, 0, I(32), ForKind::kVectorized, body, Optional<IterVar>(),
             MakeInteriorAnnotation(1));
  body = For(ki, 0, I(8), ForKind::kSerial, body, Optional<IterVar>(),
             MakeInteriorAnnotation(0));
  body = For(axis1, 0, I(1), ForKind::kSerial, body, Optional<IterVar>(),
             MakeExecutionAxisAnnotation(1));
  body = For(axis0, 0, I(4), ForKind::kSerial, body, Optional<IterVar>(),
             MakeScopeEntryAnnotations(execution_domain_axes));
  return body;
}

Stmt Make2DNarrowTileCopy(const Buffer &dst, const Buffer &src,
                          const PrimExpr &src_col_shift) {
  Var axis0("i", DataType::Int(32));
  Var axis1("j", DataType::Int(32));
  Var ki("ki", DataType::Int(32));
  Var kj("kj", DataType::Int(32));

  PrimExpr row = axis0 * I(8) + ki;
  PrimExpr dst_col = axis1 * I(16) + kj;
  PrimExpr src_col = axis1 * I(16) + src_col_shift + kj;

  Stmt body = BufferStore(dst, BufferLoad(src, {row, src_col}), {row, dst_col});
  body = For(kj, 0, I(16), ForKind::kVectorized, body, Optional<IterVar>(),
             MakeInteriorAnnotation(1));
  body = For(ki, 0, I(8), ForKind::kSerial, body, Optional<IterVar>(),
             MakeInteriorAnnotation(0));
  body = For(axis1, 0, I(2), ForKind::kSerial, body, Optional<IterVar>(),
             MakeExecutionAxisAnnotation(1));
  body = For(axis0, 0, I(4), ForKind::kSerial, body, Optional<IterVar>(),
             MakeScopeEntryAnnotationsWithTileSize({0, 1}, 8, 16));
  return body;
}

Stmt Make3DTileLoad(const Buffer &lhs, const Buffer &rhs) {
  Var batch("b", DataType::Int(32));
  Var i("i", DataType::Int(32));
  Var j("j", DataType::Int(32));
  Var ki("ki", DataType::Int(32));
  Var kj("kj", DataType::Int(32));

  PrimExpr row = i * I(2) + ki;
  PrimExpr col = j * I(128) + kj;

  Stmt body = Evaluate(BufferLoad(lhs, {batch, row, col}) +
                       BufferLoad(rhs, {batch, row, col}));
  body = For(kj, 0, I(128), ForKind::kVectorized, body, Optional<IterVar>(),
             MakeInteriorAnnotation(1));
  body = For(ki, 0, I(2), ForKind::kSerial, body, Optional<IterVar>(),
             MakeInteriorAnnotation(0));
  body = For(j, 0, I(1), ForKind::kSerial, body, Optional<IterVar>(),
             MakeExecutionAxisAnnotation(1));
  body = For(i, 0, I(1), ForKind::kSerial, body, Optional<IterVar>(),
             MakeScopeEntryAnnotations({0, 1}));
  body = For(batch, 0, I(1), ForKind::kSerial, body);
  return body;
}

PrimFunc MakePrimFunc(const Stmt &body) { return PrimFunc(Array<Var>{}, body); }

std::vector<std::string> RegionMins(const BufferRegion &region) {
  std::vector<std::string> result;
  for (const Range &range : region->region) {
    result.push_back(PrimExprToString(range->min));
  }
  return result;
}

std::vector<std::string> RegionExtents(const BufferRegion &region) {
  std::vector<std::string> result;
  for (const Range &range : region->region) {
    result.push_back(PrimExprToString(range->extent));
  }
  return result;
}

BufferRegion FindBufferRegionByName(const Array<BufferRegion> &regions,
                                    const std::string &buffer_name) {
  for (const BufferRegion &region : regions) {
    if (region->buffer->name == buffer_name) {
      return region;
    }
  }
  return BufferRegion();
}

const TileScopeDependenceEdge *FindEdge(const TileScopeWindowGraph &graph,
                                        int src, int dst,
                                        TileScopeDependenceKind kind) {
  for (const TileScopeDependenceEdge &edge : graph.edges) {
    if (edge.src_region_index == src && edge.dst_region_index == dst &&
        edge.kind == kind) {
      return &edge;
    }
  }
  return nullptr;
}

TEST(SunmmioTileLoopFusionDiscoveryTest,
     SingleRegionBuildsOneWindowWithExpectedExternalBoundaries) {
  Buffer a_shared = MakeSharedBuffer("A_shared", {I(32), I(32)});
  Buffer b_shared = MakeSharedBuffer("B_shared", {I(32), I(32)});

  SunmmioTileLoopFusionProgram program = BuildSunmmioTileLoopFusionProgram(
      MakePrimFunc(Make2DTileCopy(b_shared, a_shared)));
  std::vector<SunmmioTileLoopFusionWindowProblem> problems =
      BuildSunmmioTileLoopFusionWindowProblems(program);

  ASSERT_EQ(program.regions.size(), 1U);
  ASSERT_EQ(program.region_runs.size(), 1U);
  EXPECT_EQ(program.region_runs[0].begin_region_index, 0);
  EXPECT_EQ(program.region_runs[0].num_regions, 1);
  ASSERT_EQ(problems.size(), 1U);
  EXPECT_TRUE(problems[0].graph.edges.empty());

  const TileScopeRegion &region = program.regions[0];
  EXPECT_EQ(region.root_name, "i");
  EXPECT_EQ(region.execution_loop_var_names,
            std::vector<std::string>({"i", "j"}));
  EXPECT_EQ(region.logical_execution_axis_keys,
            std::vector<std::string>({"i", "j"}));
  ASSERT_EQ(region.available_at_execution_depths.size(), 1U);
  EXPECT_EQ(region.available_at_execution_depths[0], 2);

  BufferRegion use_in = FindBufferRegionByName(region.use_in, "A_shared");
  BufferRegion def_out = FindBufferRegionByName(region.def_out, "B_shared");
  ASSERT_TRUE(use_in.defined());
  ASSERT_TRUE(def_out.defined());
  EXPECT_EQ(RegionMins(use_in), std::vector<std::string>({"i * 8", "j * 32"}));
  EXPECT_EQ(RegionExtents(use_in), std::vector<std::string>({"8", "32"}));
  EXPECT_EQ(RegionMins(def_out), std::vector<std::string>({"i * 8", "j * 32"}));
  EXPECT_EQ(RegionExtents(def_out), std::vector<std::string>({"8", "32"}));
}

TEST(SunmmioTileLoopFusionDiscoveryTest,
     TwoConsecutiveRegionsProduceOneRawDependenceEdge) {
  Buffer a_shared = MakeSharedBuffer("A_shared", {I(32), I(32)});
  Buffer tmp_shared = MakeSharedBuffer("Tmp_shared", {I(32), I(32)});
  Buffer b_shared = MakeSharedBuffer("B_shared", {I(32), I(32)});

  PrimFunc func =
      MakePrimFunc(SeqStmt(Array<Stmt>{Make2DTileCopy(tmp_shared, a_shared),
                                       Make2DTileCopy(b_shared, tmp_shared)}));
  SunmmioTileLoopFusionProgram program =
      BuildSunmmioTileLoopFusionProgram(func);
  std::vector<SunmmioTileLoopFusionWindowProblem> problems =
      BuildSunmmioTileLoopFusionWindowProblems(program);

  ASSERT_EQ(program.regions.size(), 2U);
  ASSERT_EQ(program.region_runs.size(), 1U);
  EXPECT_EQ(program.region_runs[0].num_regions, 2);
  ASSERT_EQ(problems.size(), 1U);

  const TileScopeWindowGraph &graph = problems[0].graph;
  ASSERT_EQ(graph.edges.size(), 1U);
  const TileScopeDependenceEdge &edge = graph.edges[0];
  EXPECT_EQ(edge.src_region_index, 0);
  EXPECT_EQ(edge.dst_region_index, 1);
  EXPECT_EQ(edge.kind, TileScopeDependenceKind::kRAW);
  EXPECT_EQ(edge.rho, 2);
  EXPECT_EQ(edge.weight_bytes, 1024);
}

TEST(SunmmioTileLoopFusionDiscoveryTest,
     ReadAfterReadOnSameBufferProducesNoDependenceEdges) {
  Buffer a_shared = MakeSharedBuffer("A_shared", {I(32), I(32)});

  PrimFunc func = MakePrimFunc(SeqStmt(Array<Stmt>{
      Make2DTileRead(a_shared), Make2DTileRead(a_shared, "ii", "jj")}));
  SunmmioTileLoopFusionProgram program =
      BuildSunmmioTileLoopFusionProgram(func);
  std::vector<SunmmioTileLoopFusionWindowProblem> problems =
      BuildSunmmioTileLoopFusionWindowProblems(program);

  ASSERT_EQ(program.regions.size(), 2U);
  ASSERT_EQ(program.region_runs.size(), 1U);
  EXPECT_EQ(program.region_runs[0].num_regions, 2);
  ASSERT_EQ(problems.size(), 1U);
  EXPECT_TRUE(problems[0].graph.edges.empty());
}

TEST(SunmmioTileLoopFusionDiscoveryTest,
     OverwrittenDefsKillEarlierRawAndPreserveOnlyLatestProducer) {
  Buffer a_shared = MakeSharedBuffer("A_shared", {I(32), I(32)});
  Buffer b_shared = MakeSharedBuffer("B_shared", {I(32), I(32)});
  Buffer tmp_shared = MakeSharedBuffer("Tmp_shared", {I(32), I(32)});
  Buffer c_shared = MakeSharedBuffer("C_shared", {I(32), I(32)});

  PrimFunc func =
      MakePrimFunc(SeqStmt(Array<Stmt>{Make2DTileCopy(tmp_shared, a_shared),
                                       Make2DTileCopy(tmp_shared, b_shared),
                                       Make2DTileCopy(c_shared, tmp_shared)}));
  SunmmioTileLoopFusionProgram program =
      BuildSunmmioTileLoopFusionProgram(func);
  std::vector<SunmmioTileLoopFusionWindowProblem> problems =
      BuildSunmmioTileLoopFusionWindowProblems(program);

  ASSERT_EQ(program.regions.size(), 3U);
  ASSERT_EQ(problems.size(), 1U);

  const TileScopeWindowGraph &graph = problems[0].graph;
  ASSERT_EQ(graph.edges.size(), 2U);

  const TileScopeDependenceEdge *waw =
      FindEdge(graph, 0, 1, TileScopeDependenceKind::kWAW);
  const TileScopeDependenceEdge *raw =
      FindEdge(graph, 1, 2, TileScopeDependenceKind::kRAW);
  ASSERT_NE(waw, nullptr);
  ASSERT_NE(raw, nullptr);
  EXPECT_EQ(waw->rho, 2);
  EXPECT_EQ(waw->weight_bytes, 0);
  EXPECT_EQ(raw->rho, 2);
  EXPECT_EQ(raw->weight_bytes, 1024);
  EXPECT_EQ(FindEdge(graph, 0, 2, TileScopeDependenceKind::kRAW), nullptr);
}

TEST(SunmmioTileLoopFusionDiscoveryTest, RecordsWarEdgesForReadThenOverwrite) {
  Buffer a_shared = MakeSharedBuffer("A_shared", {I(32), I(32)});
  Buffer tmp_shared = MakeSharedBuffer("Tmp_shared", {I(32), I(32)});
  Buffer b_shared = MakeSharedBuffer("B_shared", {I(32), I(32)});

  PrimFunc func =
      MakePrimFunc(SeqStmt(Array<Stmt>{Make2DTileCopy(b_shared, tmp_shared),
                                       Make2DTileCopy(tmp_shared, a_shared)}));
  SunmmioTileLoopFusionProgram program =
      BuildSunmmioTileLoopFusionProgram(func);
  std::vector<SunmmioTileLoopFusionWindowProblem> problems =
      BuildSunmmioTileLoopFusionWindowProblems(program);

  ASSERT_EQ(program.regions.size(), 2U);
  ASSERT_EQ(problems.size(), 1U);
  const TileScopeWindowGraph &graph = problems[0].graph;
  ASSERT_EQ(graph.edges.size(), 1U);

  const TileScopeDependenceEdge &edge = graph.edges[0];
  EXPECT_EQ(edge.src_region_index, 0);
  EXPECT_EQ(edge.dst_region_index, 1);
  EXPECT_EQ(edge.kind, TileScopeDependenceKind::kWAR);
  EXPECT_EQ(edge.rho, 2);
  EXPECT_EQ(edge.weight_bytes, 0);
}

TEST(SunmmioTileLoopFusionDiscoveryTest,
     SymbolicRawOverlapFallsBackToMinStaticTileExtents) {
  Buffer a_shared = MakeSharedBuffer("A_shared", {I(32), I(32)});
  Buffer tmp_shared = MakeSharedBuffer("Tmp_shared", {I(32), I(32)});
  Buffer b_shared = MakeSharedBuffer("B_shared", {I(32), I(32)});
  Var shift("shift", DataType::Int(32));

  PrimFunc func(
      Array<Var>{shift},
      SeqStmt(Array<Stmt>{Make2DNarrowTileCopy(tmp_shared, a_shared, I(0)),
                          Make2DNarrowTileCopy(b_shared, tmp_shared, shift)}));
  SunmmioTileLoopFusionProgram program =
      BuildSunmmioTileLoopFusionProgram(func);
  std::vector<SunmmioTileLoopFusionWindowProblem> problems =
      BuildSunmmioTileLoopFusionWindowProblems(program);

  ASSERT_EQ(program.regions.size(), 2U);
  ASSERT_EQ(problems.size(), 1U);
  const TileScopeWindowGraph &graph = problems[0].graph;
  ASSERT_EQ(graph.edges.size(), 1U);

  const TileScopeDependenceEdge &edge = graph.edges[0];
  EXPECT_EQ(edge.src_region_index, 0);
  EXPECT_EQ(edge.dst_region_index, 1);
  EXPECT_EQ(edge.kind, TileScopeDependenceKind::kRAW);
  EXPECT_EQ(edge.rho, 2);
  EXPECT_EQ(edge.weight_bytes, 512);
}

TEST(SunmmioTileLoopFusionDiscoveryTest,
     Restricts3DExternalRegionsToExposedPrefix) {
  Buffer a_shared = MakeSharedBuffer("A_shared", {I(1), I(2), I(128)});
  Buffer b_shared = MakeSharedBuffer("B_shared", {I(1), I(2), I(128)});

  SunmmioTileLoopFusionProgram program = BuildSunmmioTileLoopFusionProgram(
      MakePrimFunc(Make3DTileLoad(a_shared, b_shared)));

  ASSERT_EQ(program.regions.size(), 1U);
  const TileScopeRegion &region = program.regions[0];
  EXPECT_EQ(region.root_name, "i");
  EXPECT_EQ(region.execution_loop_var_names,
            std::vector<std::string>({"i", "j"}));
  EXPECT_TRUE(region.available_at_execution_depths.empty());

  BufferRegion lhs = FindBufferRegionByName(region.use_in, "A_shared");
  BufferRegion rhs = FindBufferRegionByName(region.use_in, "B_shared");
  ASSERT_TRUE(lhs.defined());
  ASSERT_TRUE(rhs.defined());
  EXPECT_EQ(RegionMins(lhs),
            std::vector<std::string>({"b", "i * 2", "j * 128"}));
  EXPECT_EQ(RegionExtents(lhs), std::vector<std::string>({"1", "2", "128"}));
}

TEST(SunmmioTileLoopFusionDiscoveryTest,
     SwappedExecutionDomainMapsLogicalAxes) {
  Buffer a_shared = MakeSharedBuffer("A_shared", {I(32), I(32)});
  Buffer b_shared = MakeSharedBuffer("B_shared", {I(32), I(32)});

  SunmmioTileLoopFusionProgram program = BuildSunmmioTileLoopFusionProgram(
      MakePrimFunc(Make2DTileCopy(b_shared, a_shared, "j", "i", {1, 0})));

  ASSERT_EQ(program.regions.size(), 1U);
  const TileScopeRegion &region = program.regions[0];
  EXPECT_EQ(region.root_name, "j");
  EXPECT_EQ(region.execution_loop_var_names,
            std::vector<std::string>({"j", "i"}));
  EXPECT_EQ(region.logical_execution_axis_keys,
            std::vector<std::string>({"j", "i"}));
  ASSERT_EQ(region.available_at_execution_depths.size(), 1U);
  EXPECT_EQ(region.available_at_execution_depths[0], 2);
}

} // namespace
} // namespace tl
} // namespace tvm
