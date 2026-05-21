/*!
 * \file legalize_sunmmio_gemm.cc
 * \brief Legalize Sunmmio bf16 GEMM into the two-pass form required by the
 *        bf16 tensor core.
 *
 * ─── Background ────────────────────────────────────────────────────────────
 * The bf16 MMA on the Sunmmio A4E target can only read from ASRAM's north
 * bank. A single transfer fills ASRAM end-to-end, but only the data that
 * happens to land in north stripes (1 KiB-aligned, alternating with south)
 * is visible to TC. To process the full A operand, the operand's writer and
 * the GEMM itself must each be invoked twice: once on the original data, and
 * once on data re-staged so that what previously landed in south now lands
 * in north.
 *
 * ─── Pass architecture (4 phases) ──────────────────────────────────────────
 * 1. Reaching definitions analysis (Bf16AsramOperandFlow):
 *    structured RD computation over SeqStmt / For / If / Block. Output:
 *    list of (gemm, reaching writer) records. Handles nested loops and
 *    if-then-else branches uniformly via a join operation.
 *
 * 2. Validate: reject NONE (no reaching writer — typically conditional
 *    writer doesn't dominate the gemm) and MULTIPLE (e.g., divergent
 *    writers in if/else branches with no per-branch handling yet).
 *    Reject unsupported writer op types. All rejections produce fatal
 *    diagnostics with the offending buffer name.
 *
 * 3. Plan: per-gemm classify the action — InPlace, InPlaceGroup,
 *    ReissueFromSource, or ReissueFromShadow — from the writer's scope path vs
 * the gemm's, the number of consumers sharing the A operand, and (for the
 *    reissue cases) the writer-source cleanliness.
 *
 *    - InPlace: co-scoped writer, single consumer. The writer stays and
 *      restores stripe-0 every iteration for free.
 *    - InPlaceGroup: co-scoped writer, multiple consumers that form a
 *      consecutive run of siblings in one SeqStmt. The writer stays; the
 *      whole run is emitted as one [G1..Gn, W', G1'..Gn'] block.
 *    - ReissueFromSource: the writer is removed and re-issued per gemm
 *      (restore + restage) reading its source directly — source provably
 *      clean (SourceBufferCleanliness). Covers a hoisted single writer and
 *      a non-groupable multi-consumer alike.
 *    - ReissueFromShadow: like ReissueFromSource, but the source is contested,
 * so a private RSRAM stage buffer (co-located with A_dist) snapshots the source
 * and restore/restage read the snapshot.
 *
 *    A non-groupable multi-consumer A operand emits a warning — each
 *    consumer is legalized independently, adding writer traffic.
 *
 * 4. Rewrite: a single StmtMutator expands each planned gemm into the
 *    two-pass form. Every loop iteration must run
 *      restore A_dist→stripe-0, gemm pass-1,
 *      restage A_dist→stripe-1, gemm pass-2.
 *    InPlace gets the stripe-0 restore for free — its co-scoped original
 *    writer runs every iteration right before the gemm — so it only
 *    appends (restage, gemm-clone). InPlaceGroup likewise keeps the writer
 *    and emits one shared restage for the whole run. The Reissue actions
 *    run their original writer only once, so the rewriter removes it and
 *    emits BOTH the stripe-0 restore and the stripe-1 restage per gemm:
 *    omitting the restore would let iteration k+1 read the stripe-1 data
 *    left by iteration k. ReissueFromSource re-reads the source directly;
 *    ReissueFromShadow reads the shadow, and a snapshot copy replaces the
 *    removed writer to populate it.
 *
 * This pass runs between SunmmioLayoutInference and LowerTileOp so it can
 * see high-level ops (Copy, AllgatherOp, Gemm, GemmPy) directly and operate
 * on their annotations and direct fields before TileOp lowering converts
 * them to TIR intrinsics.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/node/cast.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/comm.h"
#include "../op/copy.h"
#include "../op/gemm.h"
#include "../op/gemm_py.h"
#include "../op/utils.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

// ─── Helpers shared across phases ──────────────────────────────────────────

// Returns the destination Buffer for a Copy or AllgatherOp call, or
// std::nullopt otherwise. Both ops carry the destination region as args[1].
std::optional<Buffer> GetDstBuffer(const CallNode *call) {
  if (call == nullptr)
    return std::nullopt;
  if (call->op.same_as(Copy::Get()) || call->op.same_as(AllgatherOp::Get())) {
    if (call->args.size() < 2)
      return std::nullopt;
    BufferRegion dst_br = NormalizeToBufferRegion(call->args[1]);
    return dst_br->buffer;
  }
  return std::nullopt;
}

// Returns the source Buffer for a Copy or AllgatherOp call. Used by the
// hoisted-writer rewrite path (#13/#14) so the duplicated inner-loop reads
// know what buffer to source from.
std::optional<Buffer> GetSrcBuffer(const CallNode *call) {
  if (call == nullptr)
    return std::nullopt;
  if (call->op.same_as(Copy::Get()) || call->op.same_as(AllgatherOp::Get())) {
    if (call->args.empty())
      return std::nullopt;
    BufferRegion src_br = NormalizeToBufferRegion(call->args[0]);
    return src_br->buffer;
  }
  return std::nullopt;
}

// True iff the GEMM needs bf16 two-pass legalization: its A operand is bf16
// and resides in ASRAM, AND its row count exceeds the north-bank single-pass
// capacity (cfg.bf16_gemm_single_pass_max_rows). A GEMM whose row count does
// not exceed that fits entirely in the north bank in one pass, so it needs
// no duplication. Works for both Gemm and GemmPy node types (they share the
// same a_ / m_ field convention).
template <typename NodeT>
bool ShouldLegalizeGemmNode(const NodeT *node,
                            const SunmmioTileProcessorConfig &cfg) {
  if (node == nullptr)
    return false;
  if (node->a_.scope() != kSunmmioScopeASRAM)
    return false;
  if (node->a_->dtype != DataType::BFloat(16))
    return false;
  return node->m_ > cfg.bf16_gemm_single_pass_max_rows;
}

// Info extracted from a Gemm/GemmPy call that the legalization pass needs.
// accOffsetByte is read through the typed node field, not by indexing raw
// call args, so the check survives changes to the Gemm/GemmPy ctor's
// positional layout.
struct Bf16AsramGemmInfo {
  Buffer a_dist;
  DataType c_dtype;
  int acc_offset_byte; // 0 if not yet legalized; non-zero means already done.
};

std::optional<Bf16AsramGemmInfo>
GetBf16AsramGemmInfo(const CallNode *call,
                     const SunmmioTileProcessorConfig &cfg) {
  if (call == nullptr)
    return std::nullopt;
  if (call->op.same_as(Gemm::Get())) {
    Gemm op(call->args, call->annotations);
    if (!ShouldLegalizeGemmNode(op.get(), cfg))
      return std::nullopt;
    return Bf16AsramGemmInfo{op->a_, op->c_->dtype, op->accOffsetByte_};
  }
  if (call->op.same_as(GemmPy::Get())) {
    GemmPy op(call->args, call->annotations);
    if (!ShouldLegalizeGemmNode(op.get(), cfg))
      return std::nullopt;
    return Bf16AsramGemmInfo{op->a_, op->c_->dtype, op->accOffsetByte_};
  }
  return std::nullopt;
}

// Compute the second-stripe accumulator byte offset:
//   (block_h / 2) * block_w * sizeof(C_dtype)
// block_h/block_w come from GetSunmmioLayoutBlockShape for the C dtype.
int ComputeAccOffsetBytes(Target target, DataType c_dtype) {
  Array<PrimExpr> block_shape = GetSunmmioLayoutBlockShape(target, c_dtype);
  ICHECK_EQ(block_shape.size(), 2u);
  int block_h = Downcast<IntImm>(block_shape[0])->value;
  int block_w = Downcast<IntImm>(block_shape[1])->value;
  return (block_h / 2) * block_w * c_dtype.bytes();
}

// Re-issue a writer Call with the given args and stripe offset. offset>0
// sets the src_offset_byte annotation; the op type and all other
// annotations are preserved. Shared by the direct re-issue (CloneWriter)
// and the shadow re-issue (CloneWriterFromShadow) — the two differ only in
// whether the source arg (args[0]) is replaced.
Call ReissueWriter(const CallNode *writer, Array<PrimExpr> args, int offset) {
  Map<String, ObjectRef> annotations = writer->annotations;
  if (offset != 0) {
    annotations.Set(kAttrSrcOffsetByte, IntImm(DataType::Int(32), offset));
  }
  return Call(writer->dtype, Downcast<Op>(writer->op), args, annotations);
}

// Build a compact 0-based range matching the given source region.
// Duplicated from the same-named helper in legalize_sunmmio_datapath.cc to
// keep this pass self-contained.
Array<Range> CompactRange(const Array<Range> &region) {
  Array<Range> compact;
  compact.reserve(region.size());
  for (const Range &r : region) {
    compact.push_back(Range::FromMinExtent(0, r->extent));
  }
  return compact;
}

// Allocate a fresh Buffer with the given shape in the given scope, copying
// dtype/alignment/buffer_type from `template_buf`. Used to materialize the
// ReissueFromShadow private RSRAM stage. Mirrors MakeCompactBufferWithScope in
// legalize_sunmmio_datapath.cc — duplicated locally for the same reason.
Buffer MakeShadowBufferLike(const Buffer &template_buf,
                            const Array<PrimExpr> &shape,
                            const std::string &scope, const std::string &name) {
  const auto *ptr_type =
      template_buf->data->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr_type != nullptr);
  Type new_type = PointerType(ptr_type->element_type, scope);
  Var new_var = Var(name, new_type);
  return Buffer(new_var, template_buf->dtype, shape, {}, Integer(0), name,
                template_buf->data_alignment, template_buf->offset_factor,
                template_buf->buffer_type);
}

// Clone a Gemm/GemmPy Call, setting the accOffsetByte positional arg
// (index 19 — appended after cCoords at 17/18). The Gemm and GemmPy
// constructors both read args[19] as accOffsetByte_ when present,
// defaulting to 0 otherwise.
Call CloneGemmCallWithAccOffset(const CallNode *call, int acc_offset_byte) {
  constexpr int kAccOffsetArgIndex = 19;
  Array<PrimExpr> new_args;
  new_args.reserve(std::max<size_t>(call->args.size(), kAccOffsetArgIndex + 1));
  for (size_t i = 0; i < call->args.size() && i < kAccOffsetArgIndex; ++i) {
    new_args.push_back(call->args[i]);
  }
  while (static_cast<int>(new_args.size()) < kAccOffsetArgIndex) {
    new_args.push_back(IntImm(DataType::Int(32), 0));
  }
  new_args.push_back(IntImm(DataType::Int(32), acc_offset_byte));
  for (size_t i = kAccOffsetArgIndex + 1; i < call->args.size(); ++i) {
    new_args.push_back(call->args[i]);
  }
  return Call(call->dtype, Downcast<Op>(call->op), new_args, call->annotations);
}

// ─── Phase 1: structured reaching-defs analysis ────────────────────────────

// The reaching-defs lattice element for a single buffer. NONE = no writer
// reaches this point; UNIQUE = one writer reaches; MULTIPLE = divergent
// writers reach (e.g., different writers in then/else branches).
//
// MULTIPLE is absorbing under JOIN — once a buffer's reaching def becomes
// MULTIPLE on any path, no further analysis can promote it back. This
// property is what lets us iterate For-loop bodies exactly twice (see the
// ForNode handling) instead of running to fixed point.
struct ReachingDef {
  enum class Kind { NONE, UNIQUE, MULTIPLE };
  Kind kind = Kind::NONE;
  const CallNode *writer = nullptr; // valid only if kind == UNIQUE

  static ReachingDef None() { return {Kind::NONE, nullptr}; }
  static ReachingDef Unique(const CallNode *w) { return {Kind::UNIQUE, w}; }
  static ReachingDef Multiple() { return {Kind::MULTIPLE, nullptr}; }

  static ReachingDef Join(const ReachingDef &a, const ReachingDef &b) {
    if (a.kind == Kind::NONE)
      return b;
    if (b.kind == Kind::NONE)
      return a;
    if (a.kind == Kind::UNIQUE && b.kind == Kind::UNIQUE &&
        a.writer == b.writer)
      return a;
    return Multiple();
  }

  bool operator==(const ReachingDef &o) const {
    return kind == o.kind && writer == o.writer;
  }
};

using ReachingDefMap = std::unordered_map<const BufferNode *, ReachingDef>;

// Per-gemm record produced by the analysis: which writer reaches A_dist at
// the gemm's site. Consumed by validation and planning.
struct GemmReachingDef {
  const CallNode *gemm;
  Buffer a_dist;
  DataType c_dtype;
  ReachingDef writer_rd;
};

// Collect A_dist buffers that already have a gemm consumer carrying a
// non-zero accOffsetByte. Such an operand has already been legalized —
// either by a previous run of this pass, or by a developer who wrote the
// two-pass form (split A stripes + paired gemms) by hand. accOffsetByte is
// 0 in fresh user code and non-zero only after legalization, so its
// presence on any consumer is an unambiguous "already done" marker.
//
// The pass-1 gemm of such a pair still carries accOffsetByte == 0, which is
// byte-identical to a fresh gemm; the only way to recognize it as already
// legalized is this sibling: the same A_dist is read by a non-zero
// consumer. Skipping the whole operand keeps the pass idempotent and stops
// it from double-legalizing a hand-written two-pass kernel.
std::unordered_set<const BufferNode *>
CollectLegalizedADists(const Stmt &body,
                       const SunmmioTileProcessorConfig &cfg) {
  std::unordered_set<const BufferNode *> result;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (call == nullptr)
      return;
    if (auto info = GetBf16AsramGemmInfo(call, cfg)) {
      if (info->acc_offset_byte != 0) {
        result.insert(info->a_dist.get());
      }
    }
  });
  return result;
}

class Bf16AsramOperandFlow : public StmtVisitor {
public:
  static std::vector<GemmReachingDef>
  Analyze(const Stmt &body, const SunmmioTileProcessorConfig &cfg) {
    Bf16AsramOperandFlow v(cfg);
    v.legalized_a_dists_ = CollectLegalizedADists(body, cfg);
    v.VisitStmt(body);
    return std::move(v.gemms_);
  }

private:
  explicit Bf16AsramOperandFlow(const SunmmioTileProcessorConfig &cfg)
      : cfg_(cfg) {}

  const SunmmioTileProcessorConfig &cfg_;
  ReachingDefMap state_;
  std::vector<GemmReachingDef> gemms_;
  // A_dist buffers already legalized (see CollectLegalizedADists). Gemms
  // reading these are skipped entirely.
  std::unordered_set<const BufferNode *> legalized_a_dists_;

  // For each For body: two passes are equivalent to fixed-point because
  // (1) the lattice (per-buffer: UNIQUE writer or MULTIPLE) is finite,
  // (2) MULTIPLE is absorbing under JOIN, and (3) body evaluation is
  // monotone — more polluted inputs produce more or equal pollution out.
  // Therefore the second pass's input already incorporates all back-edge
  // effects from the first pass; any further iteration produces the same
  // result.
  //
  // This argument breaks if a future change adds a non-absorbing lattice
  // value or a non-monotone analysis step. In that case, switch to a true
  // fixed-point loop here.
  void VisitStmt_(const ForNode *op) final {
    ReachingDefMap entry = state_;
    VisitStmt(op->body);
    ReachingDefMap after_iter1 = state_;
    state_ = JoinMaps(entry, after_iter1);
    VisitStmt(op->body);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    ReachingDefMap entry = state_;
    VisitStmt(op->then_case);
    ReachingDefMap then_state = state_;
    if (op->else_case.defined()) {
      state_ = entry;
      VisitStmt(op->else_case.value());
      ReachingDefMap else_state = state_;
      state_ = JoinMaps(then_state, else_state);
    } else {
      // Implicit else = no-op = entry state.
      state_ = JoinMaps(then_state, entry);
    }
  }

  void VisitStmt_(const BlockNode *op) final {
    // Block is treated as transparent for v1: descend into its body. TIR
    // Blocks carry T.reads / T.writes annotations and iter-var bindings but
    // don't introduce control-flow isolation that affects RD. Adjust if a
    // future case needs Block-induced opacity.
    VisitStmt(op->body);
  }

  void VisitStmt_(const BlockRealizeNode *op) final { VisitStmt(op->block); }

  void VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call)
      return;

    // Record the gemm reader BEFORE handling the (potential) write, so a
    // hypothetical self-writing gemm sees the prior reaching def. The two
    // sides are mutually exclusive in practice anyway (gemm ops don't write
    // to their A operand), so order is more about being explicit than
    // strictly necessary.
    //
    // A gemm is recorded only if it is not already legalized:
    //   - acc_offset_byte != 0  → this IS a pass-2 clone, skip directly;
    //   - acc_offset_byte == 0 but a_dist is in legalized_a_dists_ → this
    //     is the pass-1 gemm of an already-legalized pair (prior run, or a
    //     hand-written two-pass kernel), skip to avoid double-legalization.
    if (auto info = GetBf16AsramGemmInfo(call, cfg_)) {
      if (info->acc_offset_byte == 0 &&
          !legalized_a_dists_.count(info->a_dist.get())) {
        auto it = state_.find(info->a_dist.get());
        ReachingDef rd =
            (it != state_.end()) ? it->second : ReachingDef::None();
        gemms_.push_back({call, info->a_dist, info->c_dtype, rd});
      }
    }

    if (auto dst = GetDstBuffer(call)) {
      // Full write kills all prior defs for dst.
      state_[dst.value().get()] = ReachingDef::Unique(call);
    }
  }

  static ReachingDefMap JoinMaps(const ReachingDefMap &a,
                                 const ReachingDefMap &b) {
    ReachingDefMap result;
    for (const auto &kv : a) {
      auto it = b.find(kv.first);
      ReachingDef other = (it != b.end()) ? it->second : ReachingDef::None();
      result[kv.first] = ReachingDef::Join(kv.second, other);
    }
    for (const auto &kv : b) {
      if (result.count(kv.first))
        continue;
      result[kv.first] = ReachingDef::Join(kv.second, ReachingDef::None());
    }
    return result;
  }
};

// ─── Phase 2: validate ─────────────────────────────────────────────────────

void ValidateGemms(const std::vector<GemmReachingDef> &gemms) {
  for (const auto &g : gemms) {
    if (g.writer_rd.kind == ReachingDef::Kind::NONE) {
      LOG(FATAL) << "LegalizeSunmmioGemm: bf16-ASRAM A operand '"
                 << g.a_dist->name
                 << "' has no reaching writer at the gemm site. Likely cause: "
                    "the writer is in a conditional branch that doesn't "
                    "dominate the gemm. Move the writer outside the "
                    "conditional, or duplicate it across branches.";
    }
    if (g.writer_rd.kind == ReachingDef::Kind::MULTIPLE) {
      LOG(FATAL) << "LegalizeSunmmioGemm: bf16-ASRAM A operand '"
                 << g.a_dist->name
                 << "' has multiple reaching writers at the gemm site "
                    "(e.g., divergent writers in if/else branches). "
                    "Per-branch legalization is not yet supported.";
    }
    // The writer op type needs no validation here: Phase 1 discovers
    // writers only through GetDstBuffer, which yields a buffer exclusively
    // for Copy / AllgatherOp — so writer_rd.writer is always one of those.
  }
}

// ─── Phase 3: plan ─────────────────────────────────────────────────────────

enum class ActionKind {
  // Co-scoped writer, single consumer. The writer stays where it is and
  // restores stripe-0 every iteration for free; emit [gemm, W', gemm'].
  InPlace,
  // Co-scoped writer, multiple consumers forming a consecutive run in one
  // SeqStmt. The writer stays; emit one [G1..Gn, W', G1'..Gn'] block.
  InPlaceGroup,
  // Writer removed and re-issued per gemm (restore + restage) reading the
  // writer's source directly — source provably clean. Covers a hoisted
  // single writer and non-groupable multi-consumer alike.
  ReissueFromSource,
  // Like ReissueFromSource, but the source is contested, so restore/restage
  // read a private RSRAM snapshot instead of the source.
  ReissueFromShadow,
};

struct Action {
  ActionKind kind;
  const CallNode *writer;
  const CallNode *gemm; // primary gemm; for InPlaceGroup == group_gemms[0]
  Buffer a_dist;
  DataType c_dtype;

  // Set only when kind == InPlaceGroup: every consumer gemm of a_dist, in
  // program order. Empty for the other kinds (use `gemm`).
  std::vector<const CallNode *> group_gemms;

  // Set only when kind == ReissueFromShadow. Built in BuildPlan and consumed
  // in Phase 4. Null/empty for other kinds.
  Buffer shadow_buffer;
  Array<Range> shadow_range;
};

// Walks the body and records the enclosing IR ancestor path for each
// "interesting" call (writer or gemm). Path elements are pointers to scope-
// introducing IR nodes: ForNode and the then/else child of IfThenElseNode.
// SeqStmt and Block are transparent (no path entry). Two calls share the same
// path iff they're in the same innermost scope on the same control-flow path.
class ScopePathCollector : public StmtVisitor {
public:
  using Path = std::vector<const Object *>;

  explicit ScopePathCollector(
      const std::unordered_set<const CallNode *> &interesting)
      : interesting_(interesting) {}

  void Run(const Stmt &body) { VisitStmt(body); }

  std::unordered_map<const CallNode *, Path> paths;

private:
  const std::unordered_set<const CallNode *> &interesting_;
  Path current_path_;

  void VisitStmt_(const ForNode *op) final {
    current_path_.push_back(op);
    VisitStmt(op->body);
    current_path_.pop_back();
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    current_path_.push_back(op->then_case.get());
    VisitStmt(op->then_case);
    current_path_.pop_back();
    if (op->else_case.defined()) {
      current_path_.push_back(op->else_case.value().get());
      VisitStmt(op->else_case.value());
      current_path_.pop_back();
    }
  }

  void VisitStmt_(const BlockNode *op) final { VisitStmt(op->body); }
  void VisitStmt_(const BlockRealizeNode *op) final { VisitStmt(op->block); }

  void VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call && interesting_.count(call)) {
      paths[call] = current_path_;
    }
  }
};

// True iff `a` is a (possibly equal) prefix of `b`.
bool IsPathPrefix(const ScopePathCollector::Path &a,
                  const ScopePathCollector::Path &b) {
  if (a.size() > b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}

// Returns true iff no statement encountered AFTER `writer` (in pre-order
// traversal of `body`) writes to `source`. Writes via tile-op data-transfer
// calls are detected via GetDstBuffer; raw BufferStore is also handled.
//
// Used by Phase 3 to choose between ReissueFromSource (source clean — no shadow
// needed, re-emit reads from the source directly) and ReissueFromShadow (source
// dirty — must materialize a private RSRAM snapshot).
//
// v1 rule: any downstream write at all marks the source dirty. This is
// conservative (e.g., a kernel-scope write after the gemm's enclosing loop
// is technically safe), but it's the simplest sound rule and catches the
// flash-attention pattern cleanly.
class SourceBufferCleanliness : public StmtVisitor {
public:
  static bool Check(const Stmt &body, const CallNode *writer,
                    const Buffer &source) {
    SourceBufferCleanliness v(writer, source.get());
    v.VisitStmt(body);
    return v.clean_;
  }

private:
  SourceBufferCleanliness(const CallNode *writer, const BufferNode *source)
      : writer_(writer), source_(source) {}

  void VisitStmt_(const EvaluateNode *op) final {
    if (!clean_)
      return;
    const auto *call = op->value.as<CallNode>();
    if (call == writer_) {
      past_writer_ = true;
      return;
    }
    if (!past_writer_)
      return;
    if (auto dst = GetDstBuffer(call)) {
      if (dst.value().get() == source_)
        clean_ = false;
    }
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (!clean_ || !past_writer_)
      return;
    if (op->buffer.get() == source_)
      clean_ = false;
  }

  const CallNode *writer_;
  const BufferNode *source_;
  bool past_writer_ = false;
  bool clean_ = true;
};

// The CallNode wrapped by an Evaluate(Call) statement, or nullptr.
const CallNode *AsEvaluatedCall(const Stmt &s) {
  const auto *eval = s.as<EvaluateNode>();
  return eval ? eval->value.as<CallNode>() : nullptr;
}

// If every gemm in `consumers` is a consecutive run of siblings within one
// SeqStmt, return them in program order; otherwise std::nullopt.
//
// Adjacency is the whole groupability test. In the grouped emit
// [G1..Gn, W', G1'..Gn'] each gemm pair touches only its own C accumulator
// (pass-1 / pass-2 write disjoint stripe rows), so nothing between the
// pairs observes a half-formed C. Requiring the gemms to be strictly
// adjacent rules out any intervening statement — a stray C-read or a B-load
// between two gemms would both be reorder hazards.
std::optional<std::vector<const CallNode *>>
FindGroupableRun(const Stmt &body,
                 const std::unordered_set<const CallNode *> &consumers) {
  std::optional<std::vector<const CallNode *>> run;
  bool conflict = false;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (conflict)
      return;
    const auto *seq = node.as<SeqStmtNode>();
    if (seq == nullptr)
      return;
    std::vector<std::pair<int, const CallNode *>> hits;
    for (int i = 0; i < static_cast<int>(seq->seq.size()); ++i) {
      const CallNode *call = AsEvaluatedCall(seq->seq[i]);
      if (call != nullptr && consumers.count(call)) {
        hits.emplace_back(i, call);
      }
    }
    if (hits.empty())
      return;
    // Some consumers live in this SeqStmt — they must ALL live here, and at
    // consecutive indices.
    if (hits.size() != consumers.size()) {
      conflict = true; // spread across SeqStmts
      return;
    }
    for (size_t j = 1; j < hits.size(); ++j) {
      if (hits[j].first != hits[j - 1].first + 1) {
        conflict = true; // not adjacent
        return;
      }
    }
    std::vector<const CallNode *> ordered;
    ordered.reserve(hits.size());
    for (const auto &h : hits)
      ordered.push_back(h.second);
    run = std::move(ordered);
  });
  if (conflict)
    return std::nullopt;
  return run;
}

// The cleanliness verdict for a writer's source, plus the snapshot buffer
// to use when it is contested. `shadow` is null when `clean` is true.
struct ReissueSource {
  bool clean;
  Buffer shadow;
  Array<Range> shadow_range;
};

// Decide whether the writer's source is clean, and (if not) build the
// private RSRAM snapshot buffer the Reissue rewrite will read from. The
// snapshot's shape is the writer's source-region extents.
ReissueSource AnalyzeReissueSource(const Stmt &body, const CallNode *writer,
                                   const std::string &a_dist_name) {
  BufferRegion src = NormalizeToBufferRegion(writer->args[0]);
  ReissueSource r;
  r.clean = SourceBufferCleanliness::Check(body, writer, src->buffer);
  if (!r.clean) {
    Array<PrimExpr> shape;
    shape.reserve(src->region.size());
    for (const Range &rg : src->region)
      shape.push_back(rg->extent);
    r.shadow = MakeShadowBufferLike(src->buffer, shape, kSunmmioScopeRSRAM,
                                    a_dist_name + "_legalize_stage");
    r.shadow_range = CompactRange(src->region);
  }
  return r;
}

// ── Action factories ───────────────────────────────────────────────────
// One per ActionKind. Each fills exactly the fields its kind uses, so a
// kind can never be built with stale/irrelevant operands. BuildPlan
// constructs every Action through one of these.

// InPlace: co-scoped writer, single consumer. Writer kept.
Action MakeInPlaceAction(const GemmReachingDef &g, const CallNode *writer) {
  Action entry;
  entry.kind = ActionKind::InPlace;
  entry.writer = writer;
  entry.gemm = g.gemm;
  entry.a_dist = g.a_dist;
  entry.c_dtype = g.c_dtype;
  return entry;
}

// InPlaceGroup: co-scoped writer, a consecutive run of consumers. Writer
// kept; `group_gemms` are the consumers in program order.
Action MakeInPlaceGroupAction(const GemmReachingDef &g, const CallNode *writer,
                              std::vector<const CallNode *> group_gemms) {
  Action entry;
  entry.kind = ActionKind::InPlaceGroup;
  entry.writer = writer;
  entry.gemm = group_gemms.front();
  entry.a_dist = g.a_dist;
  entry.c_dtype = g.c_dtype;
  entry.group_gemms = std::move(group_gemms);
  return entry;
}

// Reissue: the writer is removed and re-issued per gemm (restore + restage).
// ReissueFromSource when the source is clean, ReissueFromShadow otherwise —
// `src` carries the (possibly shared) snapshot buffer.
Action MakeReissueAction(const GemmReachingDef &g, const CallNode *writer,
                         const ReissueSource &src) {
  Action entry;
  entry.kind =
      src.clean ? ActionKind::ReissueFromSource : ActionKind::ReissueFromShadow;
  entry.writer = writer;
  entry.gemm = g.gemm;
  entry.a_dist = g.a_dist;
  entry.c_dtype = g.c_dtype;
  entry.shadow_buffer = src.shadow;
  entry.shadow_range = src.shadow_range;
  return entry;
}

// True iff `first` is visited strictly before `second` in a post-order walk
// of `body`. For two distinct leaf Call statements that are not in an
// ancestor relationship, this is their textual — hence per-iteration
// execution — order. Used to confirm an InPlace writer actually precedes
// (dominates) its gemm within the scope they share.
bool StmtCallPrecedes(const Stmt &body, const CallNode *first,
                      const CallNode *second) {
  int counter = 0;
  int first_at = -1;
  int second_at = -1;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (node.get() == first)
      first_at = counter;
    else if (node.get() == second)
      second_at = counter;
    ++counter;
  });
  ICHECK(first_at >= 0 && second_at >= 0)
      << "StmtCallPrecedes: writer or gemm not found in body.";
  return first_at < second_at;
}

std::vector<Action> BuildPlan(const Stmt &body,
                              const std::vector<GemmReachingDef> &gemms,
                              const ScopePathCollector &paths) {
  // Dedup: the two-pass For traversal records loop gemms twice. Keep the
  // last (pass-2 = converged) reaching def per distinct gemm, in first-seen
  // (program) order.
  std::unordered_map<const CallNode *, GemmReachingDef> rd_of;
  std::vector<const CallNode *> order;
  for (const auto &g : gemms) {
    if (!rd_of.count(g.gemm))
      order.push_back(g.gemm);
    rd_of[g.gemm] = g;
  }

  // Consumer set per A_dist.
  std::unordered_map<const BufferNode *, std::unordered_set<const CallNode *>>
      consumers;
  for (const CallNode *gp : order) {
    consumers[rd_of[gp].a_dist.get()].insert(gp);
  }

  std::vector<Action> plan;
  std::unordered_set<const BufferNode *> a_dist_done;

  for (const CallNode *gp : order) {
    const GemmReachingDef &g = rd_of[gp];
    ICHECK_EQ(static_cast<int>(g.writer_rd.kind),
              static_cast<int>(ReachingDef::Kind::UNIQUE));
    const CallNode *writer = g.writer_rd.writer;
    const auto &consumer_set = consumers[g.a_dist.get()];

    auto wit = paths.paths.find(writer);
    ICHECK(wit != paths.paths.end());

    // ── single consumer ───────────────────────────────────────────────
    if (consumer_set.size() == 1) {
      auto git = paths.paths.find(g.gemm);
      ICHECK(git != paths.paths.end());
      bool co_scoped = (wit->second == git->second);
      // InPlace is correct only if the co-scoped writer runs *before* the
      // gemm every iteration — that running writer is the free stripe-0
      // restore. A writer that follows its gemm is, today, already rejected
      // upstream: Phase 1 records the gemm on the two-pass For walk's first
      // pass too, where its reaching def is still NONE (or MULTIPLE with a
      // pre-loop writer), and ValidateGemms rejects on that. This explicit
      // program-order check makes the InPlace precondition local and
      // robust rather than an emergent cross-phase property — and if it
      // ever does fail, Reissue (which emits its own restore) is the
      // correct fallback.
      if (co_scoped && StmtCallPrecedes(body, writer, g.gemm)) {
        plan.push_back(MakeInPlaceAction(g, writer));
      } else if (co_scoped || IsPathPrefix(wit->second, git->second)) {
        ReissueSource src = AnalyzeReissueSource(body, writer, g.a_dist->name);
        plan.push_back(MakeReissueAction(g, writer, src));
      } else {
        // Diverging paths — should have been caught upstream by
        // NONE/MULTIPLE, but defensive.
        LOG(FATAL) << "LegalizeSunmmioGemm: writer and gemm are in diverging "
                      "scope paths for bf16-ASRAM A operand '"
                   << g.a_dist->name << "'.";
      }
      continue;
    }

    // ── multi-consumer: plan the whole A_dist on first encounter ───────
    if (a_dist_done.count(g.a_dist.get()))
      continue;
    a_dist_done.insert(g.a_dist.get());

    auto run = FindGroupableRun(body, consumer_set);
    // InPlaceGroup keeps the writer, so the writer must be co-scoped with
    // the run AND precede its first gemm — same dominance requirement as
    // single-consumer InPlace. If it follows, fall through to per-consumer
    // Reissue.
    bool writer_co_scoped = run.has_value() &&
                            paths.paths.at(run->front()) == wit->second &&
                            StmtCallPrecedes(body, writer, run->front());

    if (run.has_value() && writer_co_scoped) {
      // Groupable: one [G1..Gn, W', G1'..Gn'] block, writer kept.
      plan.push_back(MakeInPlaceGroupAction(g, writer, *run));
    } else {
      // Not groupable (non-adjacent, or hoisted multi-consumer). Each
      // consumer becomes a self-restoring Reissue — always correct, but it
      // issues an extra writer pair per consumer.
      LOG(WARNING) << "LegalizeSunmmioGemm: bf16-ASRAM A operand '"
                   << g.a_dist->name << "' has " << consumer_set.size()
                   << " gemm consumers that cannot be grouped; each is "
                      "legalized independently, adding writer traffic. To "
                      "avoid this, give each consumer its own A buffer.";
      // Cleanliness is a property of the writer's source — decided once;
      // the snapshot (if any) is shared by all consumer entries so the
      // Block gets exactly one shadow allocation.
      ReissueSource src = AnalyzeReissueSource(body, writer, g.a_dist->name);
      for (const CallNode *cgp : consumer_set) {
        plan.push_back(MakeReissueAction(rd_of[cgp], writer, src));
      }
    }
  }
  return plan;
}

// ─── Phase 4: rewrite ──────────────────────────────────────────────────────

class LegalizeSunmmioGemmRewriter : public StmtExprMutator {
public:
  LegalizeSunmmioGemmRewriter(Target target,
                              const SunmmioTileProcessorConfig &cfg,
                              const std::vector<Action> &plan)
      : target_(target), cfg_(cfg) {
    for (const auto &e : plan) {
      if (e.kind == ActionKind::InPlaceGroup) {
        // Every group gemm maps to the shared entry; the SeqStmt rewriter
        // emits the whole group at the first one and skips the rest.
        for (const CallNode *g : e.group_gemms)
          gemm_plans_[g] = &e;
      } else {
        gemm_plans_[e.gemm] = &e;
      }
      // Reissue writers are removed from their original site (re-issued
      // inside the gemm's loop). ReissueFromShadow leaves a snapshot copy in
      // the writer's place to populate the shadow buffer.
      if (e.kind == ActionKind::ReissueFromSource ||
          e.kind == ActionKind::ReissueFromShadow) {
        writer_plans_[e.writer] = &e;
      }
      if (e.kind == ActionKind::ReissueFromShadow) {
        shadows_by_a_dist_[e.a_dist.get()] = e.shadow_buffer;
      }
    }
  }

private:
  Target target_;
  SunmmioTileProcessorConfig cfg_;
  std::unordered_map<const CallNode *, const Action *> gemm_plans_;
  // Hoisted writers, removed from their outer site (re-issued inside the
  // gemm's loop). ReissueFromShadow additionally leaves a snapshot copy in the
  // writer's place to populate the shadow buffer.
  std::unordered_map<const CallNode *, const Action *> writer_plans_;
  // A_dist → its shadow Buffer. Drives the alloc_buffers append in the Block
  // that owns A_dist.
  std::unordered_map<const BufferNode *, Buffer> shadows_by_a_dist_;

  // Co-locate the shadow buffer with its A_dist by appending to the same
  // Block's alloc_buffers. The shadow then inherits A_dist's liveness so
  // downstream pipelining / merge-shared see them as siblings.
  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    Array<Buffer> to_add;
    for (const Buffer &buf : block->alloc_buffers) {
      auto it = shadows_by_a_dist_.find(buf.get());
      if (it != shadows_by_a_dist_.end()) {
        to_add.push_back(it->second);
      }
    }
    if (to_add.empty())
      return block;
    Array<Buffer> alloc_buffers = block->alloc_buffers;
    for (const Buffer &b : to_add)
      alloc_buffers.push_back(b);
    block.CopyOnWrite()->alloc_buffers = std::move(alloc_buffers);
    return block;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Stmt visited = StmtExprMutator::VisitStmt_(op);
    if (!visited->IsInstance<SeqStmtNode>())
      return visited;
    SeqStmt seq = Downcast<SeqStmt>(visited);

    Array<Stmt> rewritten;
    rewritten.reserve(seq->seq.size());
    for (size_t i = 0; i < seq->seq.size(); ++i) {
      const Stmt &s = seq->seq[i];
      const CallNode *call = AsEvaluatedCall(s);

      // Gemm site: replace the gemm with its two-pass expansion.
      if (call) {
        auto git = gemm_plans_.find(call);
        if (git != gemm_plans_.end()) {
          const Action &entry = *git->second;
          if (entry.kind == ActionKind::InPlaceGroup) {
            // Emit the whole [G1..Gn, W', G1'..Gn'] block at the first
            // group gemm; the other group gemms are consecutive and are
            // skipped (consumed into the block).
            if (call == entry.group_gemms.front()) {
              EmitGroupExpansion(entry, seq->seq, i, &rewritten);
            }
          } else {
            EmitGemmExpansion(entry, s, &rewritten);
          }
          continue;
        }
      }

      // Writer site (Reissue only): the writer is re-issued inside the
      // gemm's loop, so drop it here. ReissueFromShadow leaves a snapshot copy
      // in its place to populate the shadow buffer.
      if (call) {
        auto wit = writer_plans_.find(call);
        if (wit != writer_plans_.end()) {
          if (wit->second->kind == ActionKind::ReissueFromShadow) {
            EmitSnapshotCopy(*wit->second, &rewritten);
          }
          continue;
        }
      }

      rewritten.push_back(s);
    }
    return SeqStmt::Flatten(rewritten);
  }

  // Expand a planned gemm into the two-pass form the bf16 TC requires.
  // Every loop iteration must run:
  //   restore A_dist→stripe-0, gemm pass-1, restage A_dist→stripe-1,
  //   gemm pass-2.
  // InPlace gets the stripe-0 restore for free — its original writer is
  // co-scoped right before the gemm and runs every iteration. Hoisted
  // cases must emit the restore explicitly: their original writer lives
  // outside the loop and runs only once, so without an in-loop restore
  // iteration k+1 would read the stripe-1 data left by iteration k.
  void EmitGemmExpansion(const Action &entry, const Stmt &orig_gemm,
                         Array<Stmt> *out) const {
    const int stripe = cfg_.asram_bank_stripe_bytes;

    // pass-1 prologue: hoisted cases restore A_dist to stripe-0.
    if (entry.kind == ActionKind::ReissueFromSource) {
      out->push_back(Evaluate(CloneWriter(entry.writer, /*offset=*/0)));
    } else if (entry.kind == ActionKind::ReissueFromShadow) {
      out->push_back(Evaluate(CloneWriterFromShadow(entry, /*offset=*/0)));
    }

    // pass-1: the original gemm, untouched.
    out->push_back(orig_gemm);

    // restage A_dist to stripe-1.
    if (entry.kind == ActionKind::ReissueFromShadow) {
      out->push_back(Evaluate(CloneWriterFromShadow(entry, /*offset=*/stripe)));
    } else {
      // InPlace and ReissueFromSource re-issue the writer with the offset.
      out->push_back(Evaluate(CloneWriter(entry.writer, /*offset=*/stripe)));
    }

    // pass-2: cloned gemm carrying accOffsetByte.
    int acc_off = ComputeAccOffsetBytes(target_, entry.c_dtype);
    out->push_back(Evaluate(CloneGemmCallWithAccOffset(entry.gemm, acc_off)));
  }

  // Expand an InPlaceGroup of consumers G1..Gn (a consecutive run starting
  // at seq[start]) into [G1..Gn, W', G1'..Gn']. The co-scoped writer stays
  // where it is and restores stripe-0 for the whole pass-1 run; one shared
  // restage W' then flips A_dist to stripe-1 for the whole pass-2 run.
  void EmitGroupExpansion(const Action &entry, const Array<Stmt> &seq,
                          size_t start, Array<Stmt> *out) const {
    const size_t n = entry.group_gemms.size();

    // pass-1: the original gemms, in program order.
    for (size_t j = 0; j < n; ++j) {
      ICHECK(start + j < seq.size());
      ICHECK(AsEvaluatedCall(seq[start + j]) == entry.group_gemms[j])
          << "InPlaceGroup gemms are not the expected consecutive run";
      out->push_back(seq[start + j]);
    }

    // one restage of A_dist to stripe-1, shared by the whole group.
    out->push_back(
        Evaluate(CloneWriter(entry.writer, cfg_.asram_bank_stripe_bytes)));

    // pass-2: the cloned gemms, each with its own accOffsetByte (a group's
    // gemms may differ in C dtype).
    for (size_t j = 0; j < n; ++j) {
      const CallNode *g = entry.group_gemms[j];
      auto info = GetBf16AsramGemmInfo(g, cfg_);
      ICHECK(info.has_value());
      int acc_off = ComputeAccOffsetBytes(target_, info->c_dtype);
      out->push_back(Evaluate(CloneGemmCallWithAccOffset(g, acc_off)));
    }
  }

  // A clone of the original writer call reading its original source.
  // offset==0 reproduces the writer verbatim (the stripe-0 restore);
  // offset>0 adds src_offset_byte (the stripe-1 restage). Preserves the
  // writer's op type (Copy / Allgather).
  Call CloneWriter(const CallNode *writer, int offset) const {
    if (offset == 0)
      return tvm::ffi::GetRef<Call>(writer);
    return ReissueWriter(writer, writer->args, offset);
  }

  // Re-issue the writer with its source (args[0]) replaced by the
  // ReissueFromShadow stage buffer. Preserves the writer's op type and every
  // trailing arg — so a Copy writer becomes Copy(shadow → A_dist) and an
  // AllgatherOp writer becomes Allgather(shadow → A_dist, direction, axis,
  // ...), re-running the HLink broadcast from the snapshotted source.
  // offset==0 is the stripe-0 restore; offset>0 the stripe-1 restage.
  Call CloneWriterFromShadow(const Action &entry, int offset) const {
    Array<PrimExpr> args = entry.writer->args;
    args.Set(0, MakeRegionExpr(entry.shadow_buffer, entry.shadow_range, 1));
    return ReissueWriter(entry.writer, args, offset);
  }

  // ReissueFromShadow writer-site replacement: snapshot the writer's source
  // region into the private RSRAM shadow buffer. Runs once, outside the
  // loop, in the place the original writer occupied.
  void EmitSnapshotCopy(const Action &entry, Array<Stmt> *out) const {
    BufferRegion src_br = NormalizeToBufferRegion(entry.writer->args[0]);
    PrimExpr src_region = MakeRegionExpr(src_br->buffer, src_br->region, 1);
    PrimExpr shadow_write =
        MakeRegionExpr(entry.shadow_buffer, entry.shadow_range, 2);
    out->push_back(Evaluate(
        Call(DataType::Handle(), Copy::Get(), {src_region, shadow_write}, {})));
  }
};

// ─── Pass driver ───────────────────────────────────────────────────────────

PrimFunc Run(PrimFunc f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.defined() || !TargetIsSunmmio(target.value())) {
    return f;
  }

  SunmmioTileProcessorConfig cfg =
      GetSunmmioTileProcessorConfig(target.value());

  // Phase 1: structured reaching-defs analysis.
  std::vector<GemmReachingDef> gemms =
      Bf16AsramOperandFlow::Analyze(f->body, cfg);
  if (gemms.empty())
    return f;

  // Phase 2: validate. Rejects NONE / MULTIPLE / unsupported writer types.
  ValidateGemms(gemms);

  // Collect scope paths for each writer and gemm we're acting on, so the
  // planner can classify by scope-path comparison rather than IR-shape walk.
  std::unordered_set<const CallNode *> interesting;
  for (const auto &g : gemms) {
    interesting.insert(g.gemm);
    interesting.insert(g.writer_rd.writer);
  }
  ScopePathCollector path_collector(interesting);
  path_collector.Run(f->body);

  // Phase 3: per-gemm action plan.
  std::vector<Action> plan = BuildPlan(f->body, gemms, path_collector);

  // Phase 4: drive the rewrite from the plan.
  LegalizeSunmmioGemmRewriter rewriter(target.value(), cfg, plan);
  PrimFuncNode *fp = f.CopyOnWrite();
  fp->body = rewriter(f->body);
  return f;
}

} // namespace

Pass LegalizeSunmmioGemm() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return Run(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeSunmmioGemm", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeSunmmioGemm",
                        LegalizeSunmmioGemm);
}

} // namespace tl
} // namespace tvm
