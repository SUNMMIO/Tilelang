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
 * ─── Pass architecture ─────────────────────────────────────────────────────
 * 0. Stage supported Put/Broadcast through full-line Allgather into RSRAM
 *    and source-slot Copy. The opt-in two-communication strategy instead
 *    gathers directly into ASRAM slots before each GEMM pass.
 *
 * The remaining bf16 ASRAM GEMMs follow four phases:
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
 * see high-level ops (Copy, AllgatherOp, BroadcastOp, Gemm, GemmPy) directly
 * and operate on their annotations and direct fields before TileOp lowering
 * converts them to TIR intrinsics.
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

#include "../layout/cute_layout.h"
#include "../op/builtin.h"
#include "../op/comm.h"
#include "../op/copy.h"
#include "../op/fill.h"
#include "../op/gemm.h"
#include "../op/gemm_py.h"
#include "../op/utils.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "common/attr.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

// Data-transfer ops carry the destination region as args[1]. Put is tracked
// conservatively as a write; the guarded-put rewriter below converts its
// supported single-consumer form to the same full-mesh gather protocol used
// for Broadcast.
std::optional<Buffer> GetDstBuffer(const CallNode *call) {
  if (call == nullptr)
    return std::nullopt;
  if (call->op.same_as(Copy::Get()) || call->op.same_as(AllgatherOp::Get()) ||
      call->op.same_as(BroadcastOp::Get()) || call->op.same_as(PutOp::Get())) {
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

bool IsHardwareBatchedRegion(const PrimExpr &region_expr) {
  BufferRegion region = NormalizeToBufferRegion(region_expr);
  if (region->region.size() != 3)
    return false;
  const auto *batch = region->region[0]->extent.as<IntImmNode>();
  ICHECK(batch) << "Sunmmio batch GEMM requires a static batch extent, got "
                << region->region[0]->extent;
  return batch->value > 1;
}

bool IsBatchReductionGemm(const CallNode *call) {
  if (!call ||
      (!call->op.same_as(Gemm::Get()) && !call->op.same_as(GemmPy::Get())))
    return false;
  ICHECK_GE(call->args.size(), 10U)
      << "Sunmmio GEMM call is missing clear_accum";
  BufferRegion c_region = NormalizeToBufferRegion(call->args[2]);
  return c_region->region.size() == 2 &&
         (IsHardwareBatchedRegion(call->args[0]) ||
          IsHardwareBatchedRegion(call->args[1]));
}

Call CloneGemmCallWithClearAccum(const CallNode *call, PrimExpr clear_accum) {
  Array<PrimExpr> args = call->args;
  args.Set(9, std::move(clear_accum));
  return Call(call->dtype, Downcast<Op>(call->op), args, call->annotations);
}

class BatchReductionAccumulatorRewriter : public StmtExprMutator {
  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!IsBatchReductionGemm(call))
      return StmtExprMutator::VisitStmt_(op);

    PrimExpr clear_accum = call->args[9];
    Call accumulating_gemm = CloneGemmCallWithClearAccum(call, const_false());
    if (is_zero(clear_accum))
      return Evaluate(accumulating_gemm);

    BufferRegion c_region = NormalizeToBufferRegion(call->args[2]);
    PrimExpr zero = make_zero(c_region->buffer->dtype);
    Stmt clear = Evaluate(
        Call(DataType::Handle(), Fill::Get(), {call->args[2], zero}, {}));
    if (is_one(clear_accum))
      return SeqStmt({clear, Evaluate(accumulating_gemm)});

    ICHECK(clear_accum.dtype().is_bool())
        << "Sunmmio GEMM clear_accum must have bool dtype, got "
        << clear_accum.dtype();
    return SeqStmt(
        {IfThenElse(clear_accum, clear), Evaluate(accumulating_gemm)});
  }
};

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
  ReachingDef b_writer_rd;
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
        Buffer b = NormalizeToBufferRegion(call->args[1])->buffer;
        auto bit = state_.find(b.get());
        ReachingDef b_rd =
            (bit != state_.end()) ? bit->second : ReachingDef::None();
        gemms_.push_back({call, info->a_dist, info->c_dtype, rd, b_rd});
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

std::optional<PrimExpr> FindKernelCoreId(const Stmt &body) {
  std::optional<PrimExpr> core_id;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    const auto *attr = node.as<AttrStmtNode>();
    if (!attr || attr->attr_key != tir::attr::thread_extent)
      return;
    const auto *iter = attr->node.as<IterVarNode>();
    if (iter && iter->thread_tag == "blockIdx.x")
      core_id = iter->var;
  });
  return core_id;
}

void ValidateGemms(const std::vector<GemmReachingDef> &gemms, const Stmt &body,
                   Target target) {
  auto core_id = FindKernelCoreId(body);
  int ncols = GetSunmmioMeshConfig(target).ncol;
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
    const CallNode *writer = g.writer_rd.writer;
    ICHECK(!writer->op.same_as(PutOp::Get()))
        << "LegalizeSunmmioGemm: put writer for bf16 ASRAM A is not "
           "supported: a single destination core does not cover every "
           "GEMM core, and moving the reissued put into a guarded GEMM "
           "can skip its source core. Broadcast or copy from a complete "
           "RSRAM receive buffer instead";
    if (!writer->op.same_as(BroadcastOp::Get()))
      continue;

    BroadcastOp broadcast(writer->args, writer->annotations);
    BufferRegion src = NormalizeToBufferRegion(writer->args[0]);
    BufferRegion dst = NormalizeToBufferRegion(writer->args[1]);
    BufferRegion a = NormalizeToBufferRegion(g.gemm->args[0]);
    ICHECK_EQ(broadcast->direction, 0)
        << "LegalizeSunmmioGemm: broadcast writer for bf16 ASRAM A must be "
           "horizontal; other directions are not supported";
    arith::Analyzer analyzer;
    PrimExpr cols = IntImm(DataType::Int(32), ncols);
    ICHECK(core_id.has_value() &&
           analyzer.CanProveEqual(floordiv(broadcast->src_core, cols),
                                  floordiv(core_id.value(), cols)))
        << "LegalizeSunmmioGemm: horizontal broadcast source must be "
           "in the GEMM core's row";
    ICHECK_EQ(broadcast->size->value, -1)
        << "LegalizeSunmmioGemm: broadcast writer must transfer its full "
           "source region";
    ICHECK_EQ(src->region.size(), dst->region.size());
    ICHECK_EQ(dst->region.size(), a->region.size());
    for (size_t i = 0; i < dst->region.size(); ++i) {
      ICHECK(analyzer.CanProveEqual(dst->region[i]->min, a->region[i]->min) &&
             analyzer.CanProveEqual(dst->region[i]->extent,
                                    a->region[i]->extent) &&
             analyzer.CanProveEqual(src->region[i]->extent,
                                    dst->region[i]->extent))
          << "LegalizeSunmmioGemm: broadcast writer must cover the full "
             "GEMM A region with an equally sized source";
    }
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
  std::unordered_set<const CallNode *> conditional_calls;

private:
  const std::unordered_set<const CallNode *> &interesting_;
  Path current_path_;
  int conditional_depth_ = 0;

  void VisitStmt_(const ForNode *op) final {
    current_path_.push_back(op);
    VisitStmt(op->body);
    current_path_.pop_back();
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    ++conditional_depth_;
    current_path_.push_back(op->then_case.get());
    VisitStmt(op->then_case);
    current_path_.pop_back();
    if (op->else_case.defined()) {
      current_path_.push_back(op->else_case.value().get());
      VisitStmt(op->else_case.value());
      current_path_.pop_back();
    }
    --conditional_depth_;
  }

  void VisitStmt_(const BlockNode *op) final { VisitStmt(op->body); }
  void VisitStmt_(const BlockRealizeNode *op) final { VisitStmt(op->block); }

  void VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call && interesting_.count(call)) {
      paths[call] = current_path_;
      if (conditional_depth_ != 0)
        conditional_calls.insert(call);
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

// Keep contributor slots outside the two tiled MMA dimensions. The singleton
// axis distinguishes this storage container from a rank-3 batched GEMM; codegen
// indexes away both leading axes to produce a complete 2D MMA view.
Array<Range> CollectiveAsramRegion(const Array<Range> &tile, int peers) {
  return {Range::FromMinExtent(0, peers), Range::FromMinExtent(0, 1), tile[0],
          tile[1]};
}

Call GemmWithA(const CallNode *gemm, PrimExpr a) {
  Array<PrimExpr> args = gemm->args;
  args.Set(0, a);
  return Call(gemm->dtype, gemm->op, args, gemm->annotations);
}

// Stage a guarded put through the same public allgather path as Broadcast.
// Every core contributes to the row-wide gather, while only the destination
// core copies the source slot directly into the user's ASRAM buffer. The
// gather stays outside the target guard; no private RSRAM receive buffer
// or gather-to-receive vector copy is needed.
class GuardedPutGemmRewriter : public StmtExprMutator {
public:
  GuardedPutGemmRewriter(const Stmt &body, Target target,
                         const SunmmioTileProcessorConfig &cfg, bool twice)
      : core_id_(FindKernelCoreId(body)), target_(target), cfg_(cfg),
        twice_(twice) {
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *load = node.as<BufferLoadNode>())
        ++buffer_uses_[load->buffer.get()];
      if (const auto *store = node.as<BufferStoreNode>())
        ++buffer_uses_[store->buffer.get()];
      if (const auto *call = node.as<CallNode>()) {
        if (auto info = GetBf16AsramGemmInfo(call, cfg_)) {
          ++consumers_[info->a_dist.get()];
          if (info->acc_offset_byte != 0)
            already_legalized_.insert(info->a_dist.get());
        }
      }
    });
  }

private:
  std::optional<PrimExpr> core_id_;
  Target target_;
  const SunmmioTileProcessorConfig &cfg_;
  bool twice_;
  std::unordered_map<const BufferNode *, int> buffer_uses_;
  std::unordered_set<const BufferNode *> retired_buffers_;
  std::unordered_map<const BufferNode *, int> consumers_;
  std::unordered_set<const BufferNode *> already_legalized_;
  std::vector<Array<Buffer>> allocations_;
  int next_id_ = 0;
  int conditional_depth_ = 0;

  Stmt VisitStmt_(const BlockNode *op) final {
    allocations_.emplace_back();
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    Array<Buffer> buffers;
    for (const Buffer &buffer : block->alloc_buffers)
      if (!retired_buffers_.count(buffer.get()))
        buffers.push_back(buffer);
    Map<Buffer, Layout> layouts;
    if (auto existing = block->annotations.Get(attr::kLayoutMap))
      layouts = Downcast<Map<Buffer, Layout>>(existing.value());
    for (const auto *buffer : retired_buffers_)
      layouts.erase(ffi::GetRef<Buffer>(buffer));
    for (const Buffer &buffer : allocations_.back()) {
      buffers.push_back(buffer);
      layouts.Set(buffer, sunmmio::MakeZZ(buffer->shape,
                                          {Integer(buffer->shape.size() - 2),
                                           Integer(buffer->shape.size() - 1)},
                                          GetSunmmioLayoutBlockShape(
                                              target_, buffer->dtype)));
    }
    allocations_.pop_back();
    block.CopyOnWrite()->alloc_buffers = std::move(buffers);
    block.CopyOnWrite()->annotations.Set(attr::kLayoutMap, layouts);
    return block;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    ++conditional_depth_;
    Stmt visited = StmtExprMutator::VisitStmt_(op);
    --conditional_depth_;
    return visited;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Stmt visited = StmtExprMutator::VisitStmt_(op);
    const auto *seq = visited.as<SeqStmtNode>();
    if (!seq)
      return visited;
    Array<Stmt> out;
    for (size_t i = 0; i < seq->seq.size(); ++i) {
      const CallNode *writer = AsEvaluatedCall(seq->seq[i]);
      const auto *guard = i + 1 < seq->seq.size()
                              ? seq->seq[i + 1].as<IfThenElseNode>()
                              : nullptr;
      if (conditional_depth_ != 0 || !writer ||
          !writer->op.same_as(PutOp::Get()) || !guard ||
          guard->else_case.defined()) {
        out.push_back(seq->seq[i]);
        continue;
      }
      const CallNode *gemm = AsEvaluatedCall(guard->then_case);
      auto info = GetBf16AsramGemmInfo(gemm, cfg_);
      if (!info || info->acc_offset_byte != 0 ||
          already_legalized_.count(info->a_dist.get()) ||
          consumers_[info->a_dist.get()] != 1 || IsBatchReductionGemm(gemm)) {
        out.push_back(seq->seq[i]);
        continue;
      }
      PutOp put(writer->args, writer->annotations);
      const auto *src_id = put->src_core.as<IntImmNode>();
      const auto *dst_id = put->dst_core.as<IntImmNode>();
      const auto *eq = guard->condition.as<EQNode>();
      arith::Analyzer analyzer;
      bool guard_matches = eq && core_id_ && dst_id &&
                           ((eq->a.same_as(core_id_.value()) &&
                             analyzer.CanProveEqual(eq->b, put->dst_core)) ||
                            (eq->b.same_as(core_id_.value()) &&
                             analyzer.CanProveEqual(eq->a, put->dst_core)));
      int ncols = GetSunmmioMeshConfig(target_).ncol;
      if (!src_id || !dst_id || src_id->value == dst_id->value ||
          (src_id && dst_id &&
           src_id->value / ncols != dst_id->value / ncols) ||
          !guard_matches || put->size->value != -1) {
        out.push_back(seq->seq[i]);
        continue;
      }
      BufferRegion src = NormalizeToBufferRegion(writer->args[0]);
      BufferRegion dst = NormalizeToBufferRegion(writer->args[1]);
      BufferRegion a = NormalizeToBufferRegion(gemm->args[0]);
      bool full = src->region.size() == 2 && dst->region.size() == 2 &&
                  a->region.size() == 2 && dst->buffer.same_as(info->a_dist);
      if (full) {
        for (size_t dim = 0; dim < dst->region.size(); ++dim) {
          full &= analyzer.CanProveEqual(dst->region[dim]->min,
                                         a->region[dim]->min) &&
                  analyzer.CanProveEqual(dst->region[dim]->extent,
                                         a->region[dim]->extent) &&
                  analyzer.CanProveEqual(src->region[dim]->extent,
                                         dst->region[dim]->extent);
        }
      }
      if (!full) {
        out.push_back(seq->seq[i]);
        continue;
      }
      ICHECK(core_id_ && !allocations_.empty());
      int peers = ncols;
      Array<Range> compact = MakeCompactRegion(src->region);
      Array<Range> gathered = compact;
      gathered.Set(0, Range::FromMinExtent(0, compact[0]->extent * peers));
      if (twice_)
        gathered = CollectiveAsramRegion(compact, peers);
      std::string name =
          info->a_dist->name + "_put_" + std::to_string(next_id_++);
      Buffer send = MakeCompactBufferLike(src->buffer, compact,
                                          kSunmmioScopeRSRAM, name + "_send");
      Buffer receive = MakeCompactBufferLike(
          src->buffer, gathered,
          twice_ ? kSunmmioScopeASRAM : kSunmmioScopeRSRAM, name + "_gather");
      allocations_.back().push_back(send);
      allocations_.back().push_back(receive);

      PrimExpr send_write = MakeRegionExpr(send, compact, 2);
      PrimExpr send_read = MakeRegionExpr(send, compact, 1);
      PrimExpr receive_write = MakeRegionExpr(receive, gathered, 2);
      Stmt clear = Evaluate(Call(DataType::Handle(), Fill::Get(),
                                 {send_write, make_zero(send->dtype)}, {}));
      Stmt prepare =
          IfThenElse(core_id_.value() == put->src_core,
                     Evaluate(Call(DataType::Handle(), Copy::Get(),
                                   {writer->args[0], send_write}, {})));
      Stmt gather = Evaluate(
          Call(DataType::Handle(), AllgatherOp::Get(),
               {send_read, receive_write, IntImm(DataType::Int(32), 0),
                IntImm(DataType::Int(32), -1),
                IntImm(DataType::Int(32), twice_ ? -1 : 0), core_id_.value()},
               {}));

      arith::Analyzer slot_analyzer;
      PrimExpr source_col = slot_analyzer.Simplify(
          floormod(put->src_core, IntImm(DataType::Int(32), ncols)));
      if (twice_) {
        ICHECK_EQ(buffer_uses_[info->a_dist.get()], 2)
            << "Two-communication guarded Put requires an A buffer used "
               "only by its writer and GEMM; use the default staged strategy";
        retired_buffers_.insert(info->a_dist.get());
        Array<Range> selected = gathered;
        selected.Set(0, Range::FromMinExtent(source_col, 1));
        Call first = GemmWithA(gemm, MakeRegionExpr(receive, selected, 1));
        out.push_back(clear);
        out.push_back(prepare);
        out.push_back(gather);
        out.push_back(IfThenElse(guard->condition, Evaluate(first)));
        out.push_back(Evaluate(ReissueWriter(AsEvaluatedCall(gather),
                                             AsEvaluatedCall(gather)->args,
                                             cfg_.asram_bank_stripe_bytes)));
        out.push_back(IfThenElse(
            guard->condition,
            Evaluate(CloneGemmCallWithAccOffset(
                first.get(), ComputeAccOffsetBytes(target_, info->c_dtype)))));
        ++i;
        continue;
      }
      Array<Range> selected = compact;
      selected.Set(0, Range::FromMinExtent(source_col * compact[0]->extent,
                                           compact[0]->extent));
      Stmt copy = Evaluate(
          Call(DataType::Handle(), Copy::Get(),
               {MakeRegionExpr(receive, selected, 1), writer->args[1]}, {}));

      // Communication is public, but the ASRAM write and GEMM remain
      // guarded by the original destination condition.
      out.push_back(clear);
      out.push_back(prepare);
      out.push_back(gather);
      out.push_back(IfThenElse(
          guard->condition, SeqStmt::Flatten(copy, guard->then_case), Stmt()));
      ++i;
    }
    return SeqStmt::Flatten(out);
  }
};

// Use the full-line multicast protocol for single-source BF16 transfers.
// Each core contributes a private slot; only the requested source slot is
// copied into the user's destination by default. With the two-communication
// option, eligible A writers gather directly into ASRAM and the normal planner
// reissues the gather before the second GEMM.
class BroadcastGatherRewriter : public StmtExprMutator {
public:
  BroadcastGatherRewriter(const Stmt &body, Target target,
                          const SunmmioTileProcessorConfig &cfg, bool twice,
                          std::unordered_set<const CallNode *> a_writers,
                          std::unordered_set<const CallNode *> b_writers)
      : core_id_(FindKernelCoreId(body)), target_(target),
        a_writers_(std::move(a_writers)), b_writers_(std::move(b_writers)) {
    if (!twice)
      return;
    auto gemms = Bf16AsramOperandFlow::Analyze(body, cfg);
    std::unordered_set<const CallNode *> interesting;
    std::unordered_map<const BufferNode *, std::unordered_set<const CallNode *>>
        consumers;
    for (const auto &g : gemms) {
      consumers[g.a_dist.get()].insert(g.gemm);
      if (g.writer_rd.kind == ReachingDef::Kind::UNIQUE &&
          g.writer_rd.writer->op.same_as(BroadcastOp::Get())) {
        interesting.insert(g.gemm);
        interesting.insert(g.writer_rd.writer);
      }
    }
    ScopePathCollector paths(interesting);
    paths.Run(body);
    for (const auto &g : gemms) {
      if (!interesting.count(g.gemm))
        continue;
      const auto *writer = g.writer_rd.writer;
      const auto &path = paths.paths.at(g.gemm);
      ICHECK(!paths.conditional_calls.count(g.gemm) &&
             paths.paths.at(writer) == path &&
             consumers[g.a_dist.get()].size() == 1)
          << "Two-communication BF16 Broadcast requires one GEMM consumer "
             "in the writer's unconditional scope; use the default staged "
             "strategy for hoisted, conditional or multiple consumers";
      int uses = 0;
      bool stored = false;
      PostOrderVisit(body, [&](const ObjectRef &node) {
        if (const auto *load = node.as<BufferLoadNode>())
          uses += load->buffer.same_as(g.a_dist);
        if (const auto *store = node.as<BufferStoreNode>())
          stored |= store->buffer.same_as(g.a_dist);
      });
      ICHECK(uses == 2 && !stored)
          << "Two-communication BF16 Broadcast requires an A buffer used "
             "only by its writer and GEMM; use the default staged strategy";
      direct_writers_.insert(writer);
      direct_gemms_[g.gemm] = writer;
      retired_buffers_.insert(g.a_dist.get());
    }
  }

private:
  std::optional<PrimExpr> core_id_;
  Target target_;
  std::vector<Array<Buffer>> allocations_;
  int conditional_depth_ = 0;
  int next_id_ = 0;
  std::unordered_set<const CallNode *> direct_writers_;
  std::unordered_map<const CallNode *, const CallNode *> direct_gemms_;
  std::unordered_map<const CallNode *, PrimExpr> direct_regions_;
  std::unordered_set<const BufferNode *> retired_buffers_;
  std::unordered_set<const CallNode *> a_writers_;
  std::unordered_set<const CallNode *> b_writers_;
  std::unordered_map<const BufferNode *, size_t> allocation_scopes_;

  Stmt VisitStmt_(const BlockNode *op) final {
    allocations_.emplace_back();
    for (const Buffer &buffer : op->alloc_buffers)
      allocation_scopes_[buffer.get()] = allocations_.size() - 1;
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    Array<Buffer> buffers;
    for (const Buffer &buffer : block->alloc_buffers)
      if (!retired_buffers_.count(buffer.get()))
        buffers.push_back(buffer);
    Map<Buffer, Layout> layouts;
    if (auto existing = block->annotations.Get(attr::kLayoutMap))
      layouts = Downcast<Map<Buffer, Layout>>(existing.value());
    for (const auto *buffer : retired_buffers_)
      layouts.erase(ffi::GetRef<Buffer>(buffer));
    for (const Buffer &buffer : allocations_.back()) {
      buffers.push_back(buffer);
      // This rewrite runs after layout inference. Register the normal SRAM
      // block layout explicitly so bank DMA can materialize each gathered slot.
      layouts.Set(buffer, sunmmio::MakeZZ(buffer->shape,
                                          {Integer(buffer->shape.size() - 2),
                                           Integer(buffer->shape.size() - 1)},
                                          GetSunmmioLayoutBlockShape(
                                              target_, buffer->dtype)));
    }
    allocations_.pop_back();
    block.CopyOnWrite()->alloc_buffers = std::move(buffers);
    block.CopyOnWrite()->annotations.Set(attr::kLayoutMap, layouts);
    return block;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    ++conditional_depth_;
    Stmt result = StmtExprMutator::VisitStmt_(op);
    --conditional_depth_;
    return result;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (auto it = direct_gemms_.find(call); it != direct_gemms_.end())
      return Evaluate(GemmWithA(call, direct_regions_.at(it->second)));
    if (!call || (!a_writers_.count(call) && !b_writers_.count(call)))
      return StmtExprMutator::VisitStmt_(op);
    BroadcastOp broadcast(call->args, call->annotations);
    if (!broadcast->src->dtype.is_bfloat16())
      return StmtExprMutator::VisitStmt_(op);
    const bool required = a_writers_.count(call);
    const bool supported_direction =
        broadcast->direction >= 0 && broadcast->direction <= 2;
    // A-side broadcasts are required to satisfy the staging contract because
    // the two-pass GEMM rewrite reissues their destination writer. B-side
    // broadcasts are optional optimizations; leave forms we cannot stage
    // unchanged for BroadcastOp::Lower instead of applying A-side assertions.
    if (!required &&
        (!core_id_ || allocations_.empty() || conditional_depth_ != 0 ||
         !supported_direction || broadcast->size->value != -1 ||
         broadcast->srcOffsetByte_ != 0))
      return StmtExprMutator::VisitStmt_(op);
    ICHECK(core_id_ && !allocations_.empty());
    ICHECK_EQ(conditional_depth_, 0)
        << "BF16 broadcast allgather staging requires an unconditional "
           "communication scope";
    ICHECK(supported_direction)
        << "BF16 broadcast staging supports a row, column, or full mesh";
    ICHECK_EQ(broadcast->size->value, -1)
        << "broadcast writer must transfer its full source region";
    ICHECK_EQ(broadcast->srcOffsetByte_, 0);
    BufferRegion src = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
    if (!required && (src->region.size() != 2u || dst->region.size() != 2u))
      return StmtExprMutator::VisitStmt_(op);
    ICHECK_EQ(src->region.size(), 2u);
    ICHECK_EQ(dst->region.size(), 2u);
    arith::Analyzer analyzer;
    for (size_t d = 0; d < 2; ++d) {
      if (!required && !analyzer.CanProveEqual(src->region[d]->extent,
                                               dst->region[d]->extent))
        return StmtExprMutator::VisitStmt_(op);
      ICHECK(analyzer.CanProveEqual(src->region[d]->extent,
                                    dst->region[d]->extent))
          << "BF16 broadcast staging requires equal source and destination "
             "shapes";
    }
    if (!required && src->buffer.scope() != kSunmmioScopeRSRAM &&
        src->buffer.scope() != "global")
      return StmtExprMutator::VisitStmt_(op);
    ICHECK(src->buffer.scope() == kSunmmioScopeRSRAM ||
           src->buffer.scope() == "global")
        << "BF16 broadcast staging requires an RSRAM or global source";

    auto mesh = GetSunmmioMeshConfig(target_);
    int peers = broadcast->direction == 0   ? mesh.ncol
                : broadcast->direction == 1 ? mesh.nrow
                                            : mesh.nrow * mesh.ncol;
    Array<Range> compact = MakeCompactRegion(src->region);
    Array<Range> gathered = compact;
    gathered.Set(0, Range::FromMinExtent(0, compact[0]->extent * peers));
    bool direct = direct_writers_.count(call);
    if (direct)
      gathered = CollectiveAsramRegion(compact, peers);
    std::string name =
        src->buffer->name + "_broadcast_" + std::to_string(next_id_++);
    Buffer send = MakeCompactBufferLike(src->buffer, compact,
                                        kSunmmioScopeRSRAM, name + "_send");
    Buffer receive = MakeCompactBufferLike(
        src->buffer, gathered, direct ? kSunmmioScopeASRAM : kSunmmioScopeRSRAM,
        name + "_gather");
    // The second Copy/GEMM may be outside the writer's block. Give staging
    // buffers the destination's lifetime, including its enclosing layout map.
    auto scope = allocation_scopes_.find(dst->buffer.get());
    if (!required && scope == allocation_scopes_.end())
      return StmtExprMutator::VisitStmt_(op);
    ICHECK(scope != allocation_scopes_.end())
        << "Broadcast GEMM staging requires a locally allocated destination";
    allocations_[scope->second].push_back(send);
    allocations_[scope->second].push_back(receive);
    PrimExpr send_write = MakeRegionExpr(send, compact, 2);
    PrimExpr send_read = MakeRegionExpr(send, compact, 1);
    PrimExpr receive_write = MakeRegionExpr(receive, gathered, 2);
    PrimExpr cols = IntImm(DataType::Int(32), mesh.ncol);
    PrimExpr source = broadcast->src_core;
    PrimExpr slot = broadcast->direction == 0   ? floormod(source, cols)
                    : broadcast->direction == 1 ? floordiv(source, cols)
                                                : source;
    Array<Range> selected = compact;
    selected.Set(
        0, Range::FromMinExtent(analyzer.Simplify(slot * compact[0]->extent),
                                compact[0]->extent));
    Stmt clear = Evaluate(Call(DataType::Handle(), Fill::Get(),
                               {send_write, make_zero(send->dtype)}, {}));
    Stmt prepare = IfThenElse(core_id_.value() == source,
                              Evaluate(Call(DataType::Handle(), Copy::Get(),
                                            {call->args[0], send_write}, {})));
    Stmt gather = Evaluate(
        Call(DataType::Handle(), AllgatherOp::Get(),
             {send_read, receive_write,
              IntImm(DataType::Int(32), broadcast->direction),
              IntImm(DataType::Int(32), -1),
              IntImm(DataType::Int(32), direct ? -1 : 0), core_id_.value()},
             {}));
    if (direct) {
      selected = gathered;
      selected.Set(0, Range::FromMinExtent(analyzer.Simplify(slot), 1));
      direct_regions_[call] = MakeRegionExpr(receive, selected, 1);
      return SeqStmt::Flatten(clear, prepare, gather);
    }
    Stmt copy = Evaluate(
        Call(DataType::Handle(), Copy::Get(),
             {MakeRegionExpr(receive, selected, 1), call->args[1]}, {}));
    if (broadcast->direction != 2) {
      PrimExpr source_line = broadcast->direction == 0 ? floordiv(source, cols)
                                                       : floormod(source, cols);
      PrimExpr current_line = broadcast->direction == 0
                                  ? floordiv(core_id_.value(), cols)
                                  : floormod(core_id_.value(), cols);
      if (!analyzer.CanProveEqual(source_line, current_line))
        copy = IfThenElse(source_line == current_line, copy);
    }
    return SeqStmt::Flatten(clear, prepare, gather, copy);
  }
};

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
    r.shadow =
        MakeCompactBufferLike(src->buffer, src->region, kSunmmioScopeRSRAM,
                              a_dist_name + "_legalize_stage");
    r.shadow_range = MakeCompactRegion(src->region);
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
    Array<Stmt> rewritten;
    rewritten.reserve(op->seq.size());
    for (size_t i = 0; i < op->seq.size(); ++i) {
      const Stmt &s = op->seq[i];
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
              EmitGroupExpansion(entry, op->seq, i, &rewritten);
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

      rewritten.push_back(StmtExprMutator::VisitStmt(s));
    }
    return SeqStmt::Flatten(rewritten);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call)
      return StmtExprMutator::VisitStmt_(op);

    auto gemm_it = gemm_plans_.find(call);
    if (gemm_it != gemm_plans_.end()) {
      const Action &entry = *gemm_it->second;
      ICHECK(entry.kind != ActionKind::InPlaceGroup)
          << "Grouped GEMMs must be direct children of a SeqStmt";
      Array<Stmt> rewritten;
      EmitGemmExpansion(entry, tvm::ffi::GetRef<Stmt>(op), &rewritten);
      return SeqStmt::Flatten(rewritten);
    }

    auto writer_it = writer_plans_.find(call);
    if (writer_it != writer_plans_.end()) {
      if (writer_it->second->kind == ActionKind::ReissueFromShadow) {
        Array<Stmt> rewritten;
        EmitSnapshotCopy(*writer_it->second, &rewritten);
        return SeqStmt::Flatten(rewritten);
      }
      return Evaluate(0);
    }

    return StmtExprMutator::VisitStmt_(op);
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
  // AllgatherOp re-runs communication from the snapshotted source.
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

PrimFunc Run(PrimFunc f, bool twice) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.defined() || !TargetIsSunmmio(target.value())) {
    return f;
  }

  SunmmioTileProcessorConfig cfg =
      GetSunmmioTileProcessorConfig(target.value());

  Stmt body = f->body;
  auto original_gemms = Bf16AsramOperandFlow::Analyze(body, cfg);
  if (!original_gemms.empty()) {
    // Select the reaching A/B broadcasts before guarded Put expansion marks
    // its GEMMs as already legalized. Unrelated BF16 communication must keep
    // its original lowering and must not inherit GEMM staging restrictions.
    std::unordered_set<const CallNode *> a_broadcasts;
    std::unordered_set<const CallNode *> b_broadcasts;
    for (const auto &g : original_gemms) {
      if (g.writer_rd.kind == ReachingDef::Kind::UNIQUE &&
          g.writer_rd.writer->op.same_as(BroadcastOp::Get()))
        a_broadcasts.insert(g.writer_rd.writer);
      if (g.b_writer_rd.kind == ReachingDef::Kind::UNIQUE &&
          g.b_writer_rd.writer->op.same_as(BroadcastOp::Get()))
        b_broadcasts.insert(g.b_writer_rd.writer);
    }
    body = GuardedPutGemmRewriter(body, target.value(), cfg, twice)(body);
    ValidateGemms(Bf16AsramOperandFlow::Analyze(body, cfg), body,
                  target.value());
    body = BroadcastGatherRewriter(body, target.value(), cfg, twice,
                                   std::move(a_broadcasts),
                                   std::move(b_broadcasts))(body);
  }

  // NPU-IR accumulates an unbatched store across batched A/W and requires
  // constant-true acc. Normalize clear semantics before bf16 stripe expansion
  // so the logical accumulator is initialized at most once.
  PrimFuncNode *fp = f.CopyOnWrite();
  fp->body = BatchReductionAccumulatorRewriter()(body);

  // Phase 1: structured reaching-defs analysis.
  std::vector<GemmReachingDef> gemms =
      Bf16AsramOperandFlow::Analyze(fp->body, cfg);
  if (gemms.empty())
    return f;

  // Phase 2: validate. Rejects NONE / MULTIPLE / unsupported writer types.
  ValidateGemms(gemms, fp->body, target.value());

  // Collect scope paths for each writer and gemm we're acting on, so the
  // planner can classify by scope-path comparison rather than IR-shape walk.
  std::unordered_set<const CallNode *> interesting;
  for (const auto &g : gemms) {
    interesting.insert(g.gemm);
    interesting.insert(g.writer_rd.writer);
  }
  ScopePathCollector path_collector(interesting);
  path_collector.Run(fp->body);

  // Phase 3: per-gemm action plan.
  std::vector<Action> plan = BuildPlan(fp->body, gemms, path_collector);

  // Phase 4: drive the rewrite from the plan.
  LegalizeSunmmioGemmRewriter rewriter(target.value(), cfg, plan);
  fp->body = rewriter(fp->body);
  return f;
}

} // namespace

Pass LegalizeSunmmioGemm() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    bool twice =
        ctx->GetConfig<Bool>(kSunmmioGemmTwoCommunications, Bool(false))
            .value();
    return Run(std::move(f), twice);
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
