/*!
 * \file boundary_overlap_analysis.h
 * \brief Independent, bounded proof for original-order WorkItem overlap.
 *
 * Residency failure is not an overlap proof.  This analysis neither shares
 * transfers nor changes the per-WorkItem reduction order.  It authorizes the
 * existing region planner to try a candidate; its hazard, collective-epoch,
 * event-coverage and backend resource checks remain mandatory.
 */
#ifndef TVM_TL_TRANSFORM_BOUNDARY_OVERLAP_ANALYSIS_H_
#define TVM_TL_TRANSFORM_BOUNDARY_OVERLAP_ANALYSIS_H_

#include "tile_residency_analysis.h"
#include "../../layout/cute_layout.h"
#include "../../target/utils.h"

namespace tvm {
namespace tl {
namespace boundary_overlap {

using namespace tir;

constexpr const char *kEligible = "tl.sunmmio.boundary_overlap.eligible";
constexpr const char *kProofVersion = "tl.sunmmio.boundary_overlap.proof_version";
constexpr const char *kReadOnlyBuffers = "tl.sunmmio.boundary_overlap.read_only_buffers";
constexpr const char *kStateBuffers = "tl.sunmmio.boundary_overlap.state_buffers";
constexpr const char *kReason = "tl.sunmmio.boundary_overlap.reject_reason";

inline bool Enabled() {
  const char *raw = std::getenv("TL_SUNMMIO_MULTIDIM_BOUNDARY_OVERLAP");
  return !raw || std::atoi(raw) != 0;
}

struct Proof {
  bool accepted{false};
  std::string reason;
  Array<Buffer> read_only_buffers;
  Array<Buffer> state_buffers;
};

inline bool Local(const Buffer &buffer) {
  return !buffer.scope().empty() && buffer.scope() != "global";
}

// The first version deliberately excludes boundary control flow.  In
// particular, a partial/conditional clear must not initialize a new version.
class BoundaryCheck : public StmtExprVisitor {
 public:
  bool valid{true};
  void VisitStmt_(const IfThenElseNode *) final { valid = false; }
  void VisitStmt_(const LetStmtNode *) final { valid = false; }
  void VisitStmt_(const ForNode *op) final {
    if (!residency::StaticInt(op->min) || !residency::StaticInt(op->extent) ||
        (op->step.has_value() && !is_one(op->step.value()))) valid = false;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    for (const PrimExpr &index : op->indices) CheckIndex(index);
    if (op->predicate.defined() && !is_one(op->predicate.value())) valid = false;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    for (const PrimExpr &index : op->indices) CheckIndex(index);
    StmtExprVisitor::VisitExpr_(op);
  }
 private:
  void CheckIndex(const PrimExpr &expr) {
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (node.as<BufferLoadNode>() || node.as<CallNode>()) valid = false;
    });
  }
};

// Enumerate a bounded scalar/vector tile initialization exactly, not by its
// bounding box.  This handles lowered tiled clear indices such as i1*32+kj,
// while rejecting step=2, duplicate axes, holes and out-of-range indices.
class StaticStoreCoverage : public StmtVisitor {
 public:
  StaticStoreCoverage(Buffer buffer, bool zero_only)
      : buffer_(std::move(buffer)), zero_only_(zero_only) {
    int64_t elements = 1;
    for (const PrimExpr &dim : buffer_->shape) {
      auto n = residency::StaticInt(dim);
      if (!n || *n <= 0 || elements > kMaxPoints / *n) {
        valid_ = false;
        return;
      }
      shape_.push_back(*n);
      elements *= *n;
    }
    covered_.resize(elements, false);
  }
  bool Full(const Stmt &stmt) {
    if (valid_) VisitStmt(stmt);
    return valid_ && !covered_.empty() &&
           std::all_of(covered_.begin(), covered_.end(), [](bool x) { return x; });
  }
 private:
  void VisitStmt_(const ForNode *op) final {
    if (!valid_ || (op->step.has_value() && !is_one(op->step.value()))) {
      valid_ = false;
      return;
    }
    auto min = residency::StaticInt(tir::Substitute(op->min, bindings_));
    auto extent = residency::StaticInt(tir::Substitute(op->extent, bindings_));
    if (!min || !extent || *extent < 0 || *extent > kMaxPoints ||
        *min > std::numeric_limits<int64_t>::max() - *extent) {
      valid_ = false;
      return;
    }
    auto saved = bindings_.Get(op->loop_var);
    for (int64_t i = 0; valid_ && i < *extent; ++i) {
      if (++visits_ > 4 * kMaxPoints) { valid_ = false; break; }
      bindings_.Set(op->loop_var, make_const(op->loop_var.dtype(), *min + i));
      VisitStmt(op->body);
    }
    if (saved.has_value()) bindings_.Set(op->loop_var, saved.value());
    else bindings_.erase(op->loop_var);
  }
  void VisitStmt_(const IfThenElseNode *) final { valid_ = false; }
  void VisitStmt_(const BufferStoreNode *op) final {
    if (!residency::SameStorage(op->buffer, buffer_)) return;
    const auto *floating = op->value.as<FloatImmNode>();
    if ((zero_only_ && !is_zero(op->value) &&
         !(floating && floating->value == 0.0)) ||
        (op->predicate.defined() && !is_one(op->predicate.value())) ||
        op->indices.size() != shape_.size()) {
      valid_ = false;
      return;
    }
    int64_t offset = 0;
    for (size_t i = 0; i < shape_.size(); ++i) {
      auto index = residency::StaticInt(tir::Substitute(op->indices[i], bindings_));
      if (!index || *index < 0 || *index >= shape_[i]) {
        valid_ = false;
        return;
      }
      offset = offset * shape_[i] + *index;
    }
    covered_[offset] = true;
  }
  static constexpr int64_t kMaxPoints = 65536;
  Buffer buffer_;
  bool zero_only_;
  bool valid_{true};
  int64_t visits_{0};
  std::vector<int64_t> shape_;
  std::vector<bool> covered_;
  Map<Var, PrimExpr> bindings_;
};

inline bool DisjointAt(const BufferRegion &left, const BufferRegion &right,
                       const For &outer, int64_t a, int64_t b) {
  if (!residency::SameStorage(left->buffer, right->buffer)) return true;
  if (left->region.size() != right->region.size()) return false;
  Map<Var, PrimExpr> lhs{{outer->loop_var, make_const(outer->loop_var.dtype(), a)}};
  Map<Var, PrimExpr> rhs{{outer->loop_var, make_const(outer->loop_var.dtype(), b)}};
  arith::Analyzer analyzer;
  for (size_t axis = 0; axis < left->region.size(); ++axis) {
    PrimExpr lb = tir::Substitute(left->region[axis]->min, lhs);
    PrimExpr le = lb + tir::Substitute(left->region[axis]->extent, lhs);
    PrimExpr rb = tir::Substitute(right->region[axis]->min, rhs);
    PrimExpr re = rb + tir::Substitute(right->region[axis]->extent, rhs);
    if (analyzer.CanProve(Or(LE(le, rb), LE(re, lb)))) return true;
  }
  return false;
}

// BufferLoad/Store-only escape checks miss `consume(acc.data)` and views with
// a distinct data Var introduced by match_buffer.  Keep this proof narrow:
// raw storage pointers outside the region are escapes, and any unlowered
// match_buffer in the function is an unmodelled alias boundary.
class WrittenStateEscapeCheck : public StmtExprVisitor {
 public:
  WrittenStateEscapeCheck(const ForNode *region, Array<Buffer> buffers)
      : region_(region), buffers_(std::move(buffers)) {}
  bool escapes{false};
  bool match_buffer_unmodeled{false};
 private:
  void VisitStmt_(const ForNode *op) final {
    if (op != region_) StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BlockNode *op) final {
    match_buffer_unmodeled = match_buffer_unmodeled || !op->match_buffers.empty();
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const VarNode *op) final {
    for (const Buffer &buffer : buffers_)
      escapes = escapes || buffer->data.get() == op;
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    escapes = escapes || residency::ContainsStorage(buffers_, op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    escapes = escapes || residency::ContainsStorage(buffers_, op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  const ForNode *region_;
  Array<Buffer> buffers_;
};

class RawStatePointerCheck : public StmtExprVisitor {
 public:
  explicit RawStatePointerCheck(Array<Buffer> buffers) : buffers_(std::move(buffers)) {}
  bool found{false};
 private:
  void VisitExpr_(const VarNode *op) final {
    for (const Buffer &buffer : buffers_) found = found || buffer->data.get() == op;
  }
  Array<Buffer> buffers_;
};

// An empty prefix is not an implicit zero initialization.  The only supported
// replacement is a complete first-K, BF16 two-stripe MMA definition.  Metadata
// and predicates need explicit checks: the default visitors/region collector
// do not visit every memory predicate or Buffer field.
class EmptyPrefixSafety : public StmtExprVisitor {
 public:
  bool valid{true};
 private:
  void CheckMetadata(const PrimExpr &expr) {
    if (!residency::StaticInt(expr)) valid = false;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (node.as<BufferLoadNode>() || node.as<ProducerLoadNode>() ||
          node.as<CallNode>()) valid = false;
    });
  }
  void CheckBuffer(const Buffer &buffer) {
    for (const PrimExpr &dim : buffer->shape) CheckMetadata(dim);
    for (const PrimExpr &stride : buffer->strides) CheckMetadata(stride);
    CheckMetadata(buffer->elem_offset);
  }
  void CheckIndices(const Array<PrimExpr> &indices) {
    for (const PrimExpr &index : indices)
      PostOrderVisit(index, [&](const ObjectRef &node) {
        if (node.as<BufferLoadNode>() || node.as<CallNode>()) valid = false;
      });
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    CheckBuffer(op->buffer);
    CheckIndices(op->indices);
    if (op->predicate.defined() && !is_one(op->predicate.value())) valid = false;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    CheckBuffer(op->buffer);
    CheckIndices(op->indices);
    if (op->predicate.defined() && !is_one(op->predicate.value())) valid = false;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BlockNode *op) final {
    if (!op->match_buffers.empty()) valid = false;
    for (const Buffer &buffer : op->alloc_buffers) CheckBuffer(buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const LetStmtNode *op) final {
    if (op->var.dtype().is_handle()) valid = false;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const LetNode *op) final {
    if (op->var.dtype().is_handle()) valid = false;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    if (!op->annotations.empty()) valid = false;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const ProducerLoadNode *) final { valid = false; }
};

inline bool FullRankTwo(const BufferRegion &region) {
  Buffer buffer = region->buffer;
  if (buffer->shape.size() != 2 || region->region.size() != 2 ||
      !buffer->strides.empty() || !is_zero(buffer->elem_offset)) return false;
  arith::Analyzer analyzer;
  for (size_t i = 0; i < 2; ++i) {
    auto size = residency::StaticInt(buffer->shape[i]);
    if (!size || *size <= 0 || !is_zero(region->region[i]->min) ||
        !analyzer.CanProveEqual(region->region[i]->extent, buffer->shape[i]))
      return false;
  }
  return true;
}

inline bool CanonicalLayout(const PrimFunc &func, const Buffer &buffer,
                            const Layout &expected) {
  auto matches = [&](const Layout &actual) {
    // CuteLayout::IsEqual checks mode descriptors only.  The base comparison
    // additionally substitutes common variables into the complete mapping;
    // this rejects stale placeholder identities and tampered forward indices.
    return actual.as<CuteLayoutNode>() && expected->IsEqual(actual.get()) &&
           actual->LayoutNode::IsEqual(expected.get());
  };
  auto layouts = func->GetAttr<Map<Buffer, Layout>>("layout_map");
  if (!layouts.has_value() || !layouts.value().count(buffer) ||
      !matches(layouts.value()[buffer])) return false;
  for (const auto &entry : layouts.value())
    if (residency::SameStorage(entry.first, buffer) && !entry.first.same_as(buffer))
      return false;
  bool valid = true;
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    const auto *block = node.as<BlockNode>();
    if (!block) return;
    auto annotation = block->annotations.Get("layout_map");
    if (!annotation.has_value()) return;
    auto local = Downcast<Map<Buffer, Layout>>(annotation.value());
    for (const auto &entry : local) {
      if (!residency::SameStorage(entry.first, buffer)) continue;
      if (!entry.first.same_as(buffer) || !matches(entry.second)) valid = false;
    }
  });
  return valid;
}

inline std::string ProveEmptyPrefix(
    const PrimFunc &func, const For &core, const Array<Buffer> &states,
    const std::vector<residency::Operation> &operations) {
  EmptyPrefixSafety safety;
  safety(func->body);
  if (!safety.valid) return "empty_prefix_hidden_access_or_alias";
  auto target = func->GetAttr<Target>("target");
  if (!target.has_value()) return "empty_prefix_target_required";
  auto mcpu = target.value()->GetAttr<String>("mcpu");
  // The target helpers intentionally return A4E defaults for unknown models.
  // Such a convenience fallback is not a hardware-coverage proof.
  if (!TargetIsSunmmio(target.value()) || !mcpu.has_value() ||
      mcpu.value() != "sunmmio-a4e") return "empty_prefix_target_unproven";
  auto cfg = GetSunmmioTileProcessorConfig(target.value());
  auto block = GetSunmmioLayoutBlockShape(target.value(), DataType::Float(32));
  auto a_block = GetSunmmioLayoutBlockShape(target.value(), DataType::BFloat(16));
  auto height = block.size() == 2 ? residency::StaticInt(block[0]) : std::nullopt;
  auto width = block.size() == 2 ? residency::StaticInt(block[1]) : std::nullopt;
  if (!height || !width || *height <= 0 || *width <= 0 || *height % 2 ||
      *height != cfg.block_height || *width != cfg.block_width ||
      cfg.bf16_gemm_single_pass_max_rows != *height / 2 ||
      !StructuralEqual()(block, a_block) ||
      *height / 2 * *width * 2 != cfg.asram_bank_stripe_bytes)
    return "empty_prefix_stripe_geometry_unproven";
  const int64_t second_offset = *height / 2 * *width * DataType::Float(32).bytes();
  arith::Analyzer analyzer;
  std::map<const VarNode *, const residency::Operation *> first;
  std::map<const VarNode *, unsigned> stripes;
  for (const Buffer &state : states) stripes.emplace(state->data.get(), 0);
  for (const residency::Operation &op : operations) {
    const BufferNode *own_state = op.transfer ? nullptr : op.writes[0]->buffer.get();
    // A clear MMA does not read its C stripe.  Every other actual state read
    // must follow both first-K stripe definitions, including transfer sources
    // and another MMA's A/B operands.  Order is retained, not reconstructed.
    for (size_t i = 0; i < op.reads.size(); ++i) {
      Buffer buffer = op.reads[i]->buffer;
      auto found = stripes.find(buffer->data.get());
      if (found == stripes.end()) continue;
      if (!op.transfer && i == 2 && buffer.get() == own_state) continue;
      if (found->second != 3) return "empty_prefix_state_read_before_init";
    }
    if (op.transfer) continue;
    if (op.call->args.size() != 7 || op.reads.size() != 3 || op.writes.size() != 1)
      return "empty_prefix_mma_form_unproven";
    Buffer a = op.reads[0]->buffer, b = op.reads[1]->buffer, c = op.writes[0]->buffer;
    if (a.scope() != "shared.asram" || b.scope() != "shared.wsram" ||
        c.scope() != "shared.rsram" || a->dtype != DataType::BFloat(16) ||
        b->dtype != DataType::BFloat(16) || c->dtype != DataType::Float(32) ||
        !FullRankTwo(op.reads[0]) || !FullRankTwo(op.reads[1]) ||
        !FullRankTwo(op.writes[0]) || !is_zero(op.call->args[3]))
      return "empty_prefix_full_mma_required";
    auto trans_b = residency::StaticInt(op.call->args[4]);
    if (!trans_b || (*trans_b != 0 && *trans_b != 1))
      return "empty_prefix_full_mma_required";
    for (size_t axis = 0; axis < c->shape.size(); ++axis) {
      auto size = residency::StaticInt(c->shape[axis]);
      int64_t alignment = axis == 0 ? *height : *width;
      if (!size || *size % alignment) return "empty_prefix_partial_physical_tile";
    }
    if (!analyzer.CanProveEqual(a->shape[0], c->shape[0]) ||
        !analyzer.CanProveEqual(a->shape[1], b->shape[*trans_b ? 1 : 0]) ||
        !analyzer.CanProveEqual(c->shape[1], b->shape[*trans_b ? 0 : 1]))
      return "empty_prefix_mma_shape_mismatch";
    Layout a_expected = sunmmio::MakeZZ(a->shape, {Integer(0), Integer(1)}, a_block);
    Layout b_expected = *trans_b
        ? sunmmio::MakeZZ(b->shape, {Integer(0), Integer(1)}, a_block)
        : sunmmio::MakeNZ(b->shape, {Integer(0), Integer(1)}, a_block);
    Layout c_expected = sunmmio::MakeZZ(c->shape, {Integer(0), Integer(1)}, block);
    if (!CanonicalLayout(func, a, a_expected) || !CanonicalLayout(func, b, b_expected) ||
        !CanonicalLayout(func, c, c_expected)) return "empty_prefix_layout_unproven";
    if (!analyzer.CanProveEqual(op.call->args[5], core->loop_var == core->min))
      return "empty_prefix_first_k_clear_required";
    auto offset = residency::StaticInt(op.call->args[6]);
    auto &mask = stripes.at(c->data.get());
    if (!offset || (*offset != 0 && *offset != second_offset) ||
        (*offset == 0 ? mask != 0 : mask != 1))
      return "empty_prefix_stripe_order_or_offset";
    if (*offset == 0) {
      first.emplace(c->data.get(), &op);
      mask = 1;
    } else {
      const auto &initial = *first.at(c->data.get());
      for (size_t i = 0; i < 3; ++i)
        if (!residency::SameRegion(initial.reads[i], op.reads[i]))
          return "empty_prefix_stripe_pair_mismatch";
      if (!StructuralEqual()(initial.call->args[4], op.call->args[4]))
        return "empty_prefix_stripe_pair_mismatch";
      mask = 3;
    }
  }
  for (const auto &entry : stripes)
    if (entry.second != 3) return "empty_prefix_incomplete_stripes";
  return "";
}

inline Proof Analyze(const PrimFunc &func, const For &outer) {
  Proof proof;
  auto reject = [&](const char *reason) { proof.reason = reason; return proof; };
  auto min = residency::StaticInt(outer->min);
  auto extent = residency::StaticInt(outer->extent);
  if (!min || !extent || *extent < 2 || *extent > 8 ||
      *min > std::numeric_limits<int64_t>::max() - *extent)
    return reject("bounded_static_outer_required");
  if (outer->kind != ForKind::kSerial || outer->thread_binding.defined() ||
      (outer->step.has_value() && !is_one(outer->step.value())))
    return reject("unit_serial_outer_required");
  if (!func->HasNonzeroAttr("tir.noalias")) return reject("global_alias_unproven");
  auto nonrestrict = func->GetAttr<Array<Var>>("tl.non_restrict_params");
  if (nonrestrict.has_value() && !nonrestrict.value().empty())
    return reject("global_alias_unproven");

  auto statements = residency::Statements(outer->body);
  For core;
  Array<Stmt> prefix, suffix;
  for (const Stmt &stmt : statements) {
    const auto *loop = stmt.as<ForNode>();
    if (loop && loop->annotations.count("num_stages")) {
      if (core.defined()) return reject("non_unique_core");
      core = ffi::GetRef<For>(loop);
    } else if (!core.defined()) prefix.push_back(stmt);
    else suffix.push_back(stmt);
  }
  if (!core.defined() || suffix.empty())
    return reject("prefix_core_suffix_required");
  auto core_min = residency::StaticInt(core->min);
  auto core_extent = residency::StaticInt(core->extent);
  if (!core_min || !core_extent || *core_extent < 1 || *core_extent > 64 ||
      *core_min > std::numeric_limits<int64_t>::max() - *core_extent)
    return reject("bounded_static_core_required");
  if (core->kind != ForKind::kSerial || core->thread_binding.defined() ||
      (core->step.has_value() && !is_one(core->step.value())))
    return reject("unit_serial_core_required");
  residency::BoundaryCheck known_effects;
  known_effects(SeqStmt::Flatten(prefix));
  known_effects(SeqStmt::Flatten(suffix));
  BoundaryCheck boundary;
  boundary(SeqStmt::Flatten(prefix));
  boundary(SeqStmt::Flatten(suffix));
  if (!known_effects.valid || !boundary.valid)
    return reject("boundary_control_or_side_effect");

  std::map<const VarNode *, const BufferNode *> views;
  bool aliases = false;
  auto record = [&](const Buffer &buffer) {
    auto item = views.emplace(buffer->data.get(), buffer.get());
    aliases = aliases || (!item.second && item.first->second != buffer.get());
  };
  PostOrderVisit(outer->body, [&](const ObjectRef &node) {
    if (const auto *load = node.as<BufferLoadNode>()) record(load->buffer);
    if (const auto *store = node.as<BufferStoreNode>()) record(store->buffer);
  });
  if (aliases) return reject("ambiguous_buffer_alias");

  std::vector<residency::Operation> operations;
  Array<Buffer> written_local;
  for (const Stmt &stmt : residency::Statements(core->body)) {
    residency::Operation op;
    if (!residency::ParseOperation(stmt, &op)) return reject("core_control_or_side_effect");
    if (op.transfer && op.async.kind == SunmmioAsyncOpKind::kBroadcast) {
      auto coverage = residency::CollectiveCoverage(op.async, func);
      if (!coverage) return reject("collective_coverage_unproven");
      op.writes[0] = coverage.value();
    }
    for (const BufferRegion &write : op.writes) {
      if (!Local(write->buffer)) return reject("core_global_write");
      if (!residency::ContainsStorage(written_local, write->buffer))
        written_local.push_back(write->buffer);
      if (!op.transfer) {
        if (write->buffer.scope() != "shared.rsram" ||
            residency::RegionUses(write, outer->loop_var))
          return reject("independent_local_state_required");
        if (!residency::ContainsStorage(proof.state_buffers, write->buffer))
          proof.state_buffers.push_back(write->buffer);
      }
    }
    if (!op.transfer) {
      arith::Analyzer analyzer;
      PrimExpr clear = op.call->args[5];
      if (!is_zero(clear) &&
          !analyzer.CanProveEqual(clear, core->loop_var == core->min))
        return reject("reduction_clear_order_unproven");
      if (op.call->args.size() == 9U && !analyzer.CanProveEqual(
              op.call->args[8], core->loop_var == core->min + core->extent - 1))
        return reject("reduction_output_order_unproven");
    }
    operations.push_back(std::move(op));
  }
  if (proof.state_buffers.empty()) return reject("no_mma_consumer");

  if (prefix.empty()) {
    std::string reason = ProveEmptyPrefix(func, core, proof.state_buffers, operations);
    if (!reason.empty()) return reject(reason.c_str());
  }

  StmtReadWriteCollector rw(func);
  rw.traverse_stmt(SeqStmt::Flatten(prefix));
  if (!rw.read_buffer_regions_.empty()) return reject("prefix_reads_state");
  for (const BufferRegion &write : rw.write_buffer_regions_) {
    if (!residency::ContainsStorage(proof.state_buffers, write->buffer))
      return reject("prefix_writes_non_state");
  }
  for (const residency::Operation &op : operations) {
    if (op.transfer || !is_zero(op.call->args[5])) continue;
    if (!StaticStoreCoverage(op.writes[0]->buffer, true).Full(SeqStmt::Flatten(prefix)))
      return reject("accumulator_not_fully_initialized");
  }

  std::vector<BufferRegion> global_writes;
  std::vector<BufferRegion> suffix_global_reads;
  Array<Buffer> suffix_defined;
  for (const Stmt &stmt : suffix) {
    rw.traverse_stmt(stmt);
    for (const BufferRegion &read : rw.read_buffer_regions_) {
      if (!Local(read->buffer)) suffix_global_reads.push_back(read);
      if (Local(read->buffer) && !residency::ContainsStorage(proof.state_buffers, read->buffer) &&
          !residency::ContainsStorage(suffix_defined, read->buffer))
        return reject("suffix_reads_carried_or_ephemeral_state");
    }
    for (const BufferRegion &write : rw.write_buffer_regions_) {
      if (residency::ContainsStorage(written_local, write->buffer))
        return reject("suffix_writes_core_storage");
      if (Local(write->buffer)) {
        if (!StaticStoreCoverage(write->buffer, false).Full(stmt))
          return reject("suffix_state_not_fully_defined");
        if (!residency::ContainsStorage(suffix_defined, write->buffer))
          suffix_defined.push_back(write->buffer);
      } else {
        residency::Operation transfer;
        if (!residency::ParseOperation(stmt, &transfer) || !transfer.transfer ||
            transfer.async.kind == SunmmioAsyncOpKind::kBroadcast)
          return reject("global_suffix_requires_direct_transfer");
        global_writes.push_back(write);
      }
    }
  }
  if (global_writes.empty()) return reject("global_output_required");
  for (const BufferRegion &write : global_writes) {
    for (const residency::Operation &op : operations)
      for (const BufferRegion &read : op.reads)
        if (residency::SameStorage(write->buffer, read->buffer))
          return reject("global_output_aliases_core_input");
    for (const BufferRegion &other : global_writes)
      for (int64_t i = 0; i < *extent; ++i)
        for (int64_t j = 0; j < *extent; ++j)
          if (i != j && !DisjointAt(write, other, outer, *min + i, *min + j))
            return reject("global_output_overlap_across_workitems");
    // Epilogues may read global memory too, e.g. residual/bias/output loads.
    // A safe write-vs-write test alone does not establish cross-WorkItem
    // independence. Check every distinct pair, including nonadjacent items.
    for (const BufferRegion &read : suffix_global_reads)
      for (int64_t i = 0; i < *extent; ++i)
        for (int64_t j = 0; j < *extent; ++j)
          if (i != j && !DisjointAt(write, read, outer, *min + i, *min + j))
            return reject("suffix_global_read_write_overlap_across_workitems");
  }

  // Scratch reads must follow a complete write in the same logical K step.
  // External read-only caches are retained, never versioned by this proof.
  std::vector<BufferRegion> definitions;
  for (const residency::Operation &op : operations) {
    if (op.transfer && residency::ContainsStorage(proof.state_buffers, op.async.dst->buffer))
      return reject("transfer_overwrites_reduction_state");
    for (const BufferRegion &read : op.reads) {
      if (!Local(read->buffer) || residency::ContainsStorage(proof.state_buffers, read->buffer))
        continue;
      if (!residency::ContainsStorage(written_local, read->buffer)) {
        if (residency::ContainsStorage(suffix_defined, read->buffer))
          return reject("suffix_changes_read_only_cache");
        if (!residency::ContainsStorage(proof.read_only_buffers, read->buffer))
          proof.read_only_buffers.push_back(read->buffer);
        continue;
      }
      bool defined = false;
      for (const BufferRegion &definition : definitions)
        defined = defined || residency::SameRegion(read, definition);
      if (!defined) return reject("scratch_read_before_full_write");
    }
    for (const BufferRegion &write : op.writes) {
      definitions.erase(std::remove_if(definitions.begin(), definitions.end(),
        [&](const BufferRegion &old) { return residency::SameStorage(old->buffer, write->buffer); }),
        definitions.end());
      definitions.push_back(write);
    }
  }
  for (const Buffer &buffer : suffix_defined)
    if (!residency::ContainsStorage(written_local, buffer)) written_local.push_back(buffer);
  std::set<const VarNode *> allocations;
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *block = node.as<BlockNode>())
      for (const Buffer &buffer : block->alloc_buffers) allocations.insert(buffer->data.get());
  });
  for (const Buffer &buffer : written_local)
    if (!allocations.count(buffer->data.get())) return reject("state_not_local_allocation");
  for (const Var &param : func->params)
    for (const Buffer &buffer : written_local)
      if (param.same_as(buffer->data)) return reject("written_state_is_function_parameter");
  RawStatePointerCheck raw_pointer(written_local);
  raw_pointer(outer->body);
  if (raw_pointer.found) return reject("raw_state_pointer_inside_region");
  WrittenStateEscapeCheck escape(outer.get(), written_local);
  escape(func->body);
  if (escape.match_buffer_unmodeled) return reject("match_buffer_alias_unmodeled");
  if (escape.escapes) return reject("written_state_escapes_region");
  proof.accepted = true;
  proof.reason = "proved_original_order_boundary_overlap";
  return proof;
}

}  // namespace boundary_overlap
}  // namespace tl
}  // namespace tvm
#endif
