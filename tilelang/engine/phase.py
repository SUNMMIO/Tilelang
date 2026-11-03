from __future__ import annotations
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from .sunmmio_candidate_selection import compare_pipeline_candidate_work
from tilelang.utils.target import target_is_sunmmio
from tvm import tir, IRModule
from tvm.target import Target
import tilelang
import tilelang.utils.target as _target_utils
from tilelang.transform import PassContext
from tilelang.contrib.nvcc import have_tma, is_hopper, have_pdl


_LOGGER = logging.getLogger(__name__)
_SUNMMIO_SHADOW_DIAGNOSTIC_ENVS = (
    "TL_SUNMMIO_ILP_PROBLEM_JSON",
    "TL_SUNMMIO_ILP_SOLUTION_JSON",
    "TL_SUNMMIO_PIPELINE_GRAPH_JSON",
    "TL_SUNMMIO_MULTIDIM_REGION_JSON",
    "TL_SUNMMIO_RESIDENCY_REPORT",
)
_sunmmio_shadow_planning_depth = 0


def sunmmio_shadow_planning_active() -> bool:
    """Whether a non-selecting candidate is being planned for analysis."""
    return _sunmmio_shadow_planning_depth > 0


@contextmanager
def _sunmmio_shadow_planning_scope():
    """Keep shadow passes out of official planner artifacts and observers."""
    global _sunmmio_shadow_planning_depth
    previous = {name: os.environ.get(name) for name in _SUNMMIO_SHADOW_DIAGNOSTIC_ENVS}
    with TemporaryDirectory(prefix="tilelang_sunmmio_shadow_") as directory:
        for name, value in previous.items():
            if value is not None:
                os.environ[name] = str(Path(directory) / (name.lower() + ".json"))
        _sunmmio_shadow_planning_depth += 1
        try:
            yield
        finally:
            _sunmmio_shadow_planning_depth -= 1
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def allow_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if (not is_cuda_target(target)) or (not have_tma(target)):
        return False
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not disable_warp_specialized


def allow_tma_and_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if not have_tma(target):
        return False
    disable_tma_lower = pass_ctx.config.get("tl.disable_tma_lower", False)
    return not disable_tma_lower and allow_warp_specialized(pass_ctx=pass_ctx, target=target)


def allow_fence_proxy(target: Target | None = None) -> bool:
    return have_tma(target)


def allow_vectorize(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    disable_vectorize = pass_ctx.config.get("tir.disable_vectorize", False)
    return not disable_vectorize


def allow_global_thread_synchronization(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_global_thread_sync = pass_ctx.config.get("tir.detect_global_barrier", False)
    return enable_global_thread_sync


def should_enable_aggressive_merge(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_aggressive_merge = bool(pass_ctx.config.get(tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE, False))
    if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
        # This is a workaround to avoid the bug in the MergeSharedMemoryAllocations pass
        # when warp specialization is enabled, as different warp threads may access different
        # buffers, but the liveness analysis is hard because we need to do pipeline.
        enable_aggressive_merge = False
    return enable_aggressive_merge


def should_force_let_inline(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_FORCE_LET_INLINE, False))


def should_enable_ast_print(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_AST_PRINT_ENABLE, False))


def should_enable_sunmmio_pipeline(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return not bool(pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_SUNMMIO_PIPELINE, False))


def get_sunmmio_pipeline_mode(pass_ctx: PassContext | None = None) -> str:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    mode = pass_ctx.config.get(tilelang.PassConfigKey.TL_SUNMMIO_PIPELINE_MODE, "greedy")
    mode = str(mode).strip().lower()
    if mode not in {
        "greedy",
        "ilp",
        "multidim_ilp",
        "cross_outer",
        "region_cross_outer",
        "hierarchical_ilp",
        "c_style_pipeline",
    }:
        raise ValueError(
            f"Invalid Sunmmio pipeline mode: {mode}. Expected one of: "
            "'greedy', 'ilp', 'multidim_ilp', 'cross_outer', 'region_cross_outer', "
            "'hierarchical_ilp', 'c_style_pipeline'."
        )
    return mode


def get_sunmmio_multidim_policy(pass_ctx: PassContext | None = None) -> str:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    policy = pass_ctx.config.get(tilelang.PassConfigKey.TL_SUNMMIO_MULTIDIM_POLICY, "auto")
    policy = str(policy).strip().lower()
    if policy not in {"forced", "auto"}:
        raise ValueError(f"Invalid Sunmmio multidim policy: {policy}. Expected one of: 'forced', 'auto'.")
    return policy


def get_sunmmio_cross_outer_policy(pass_ctx: PassContext | None = None) -> str:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    policy = pass_ctx.config.get(tilelang.PassConfigKey.TL_SUNMMIO_CROSS_OUTER_POLICY, "auto")
    policy = str(policy).strip().lower()
    if policy not in {"forced", "auto"}:
        raise ValueError(f"Invalid Sunmmio cross-outer policy: {policy}. Expected one of: 'forced', 'auto'.")
    return policy


def get_sunmmio_region_cross_outer_policy(pass_ctx: PassContext | None = None) -> str:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    policy = pass_ctx.config.get(tilelang.PassConfigKey.TL_SUNMMIO_REGION_CROSS_OUTER_POLICY, "auto")
    policy = str(policy).strip().lower()
    if policy not in {"forced", "auto"}:
        raise ValueError(f"Invalid Sunmmio region cross-outer policy: {policy}. Expected one of: 'forced', 'auto'.")
    return policy


def get_sunmmio_region_cross_outer_strategy(
    pass_ctx: PassContext | None = None,
) -> str:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    strategy = pass_ctx.config.get(
        tilelang.PassConfigKey.TL_SUNMMIO_REGION_CROSS_OUTER_STRATEGY,
        "continuous",
    )
    strategy = str(strategy).strip().lower()
    if strategy not in {"conservative", "continuous", "hierarchical_ilp"}:
        raise ValueError(
            f"Invalid Sunmmio region cross-outer strategy: {strategy}. Expected one of: 'conservative', 'continuous', 'hierarchical_ilp'."
        )
    return strategy


def should_enable_layout_visual(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE, False)
    return enabled


def should_enable_race_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK, False)
    return enabled


def get_layout_visual_formats(pass_ctx: PassContext | None = None) -> list[str]:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    formats_value = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS, "")
    if not formats_value:
        return ["txt"]

    formats_str = formats_value.strip().lower()
    valid_formats = ["txt", "png", "pdf", "svg", "all"]

    if formats_str == "all":
        return ["txt", "png", "pdf", "svg"]

    if "," in formats_str:
        formats_list = [f.strip() for f in formats_str.split(",")]
    else:
        formats_list = [formats_str]

    invalid_formats = [f for f in formats_list if f not in valid_formats]
    if invalid_formats:
        raise ValueError(
            f"Invalid formats for TL_LAYOUT_VISUALIZATION_FORMATS: {invalid_formats}. "
            f"Valid formats are: {valid_formats}. "
            f"You can choose one of the valid formats or a comma-separated list of formats.(e.g., 'txt,png,pdf')"
        )
    return formats_list


def LayoutVisual(mod: IRModule, target: Target | None = None) -> None:
    """Apply layout visualization pass if enabled."""
    if should_enable_layout_visual():
        formats = get_layout_visual_formats()
        if target is not None and target_is_sunmmio(target):
            # Sunmmio uses CuteLayouts; route to a dedicated pass so the upstream
            # fragment visualizer stays untouched (keeps merges clean).
            from tilelang.analysis.sunmmio_layout_visual import SunmmioLayoutVisual

            SunmmioLayoutVisual(formats=formats)(mod)
        else:
            tilelang.analysis.LayoutVisual(formats=formats)(mod)


def PreLowerSemanticCheck(mod: IRModule) -> None:
    """
    Check whether the module is valid before lowering. If not, raise a user-friendly error
    in Python side instead of letting the error dive into the complicated TVM/C++ stack.
    Note: This is a validation-only pipeline of passes and does not modify or return the module.
    """

    # Print AST for debugging purpose
    if should_enable_ast_print():
        tilelang.analysis.ASTPrinter()(mod)
    # Check if there are any invalid nested loops.
    tilelang.analysis.NestedLoopChecker()(mod)
    # Check if there are any invalid symbolic T.Parallel + fragment access.
    tilelang.analysis.FragmentLoopChecker()(mod)


def LowerAndLegalizeSunmmio(mod: IRModule, target: Target) -> IRModule:
    mod = tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.ValidateSunmmioFrontendSyntax()(mod)
    # Establish the parameter alias contract before lowering decisions that
    # can benefit from it. Re-running HoistNonRestrictParams later remains
    # necessary because lowering may introduce additional explicit exceptions.
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    mod = _infer_sunmmio_noalias(mod)
    mod = tilelang.transform.ResolveSunmmioMeshSymbols()(mod)
    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)

    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)

    mod = tilelang.transform.LegalizeSunmmioBatchGemmViews()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
    mod = tilelang.transform.SunmmioLayoutInference()(mod)
    mod = tilelang.transform.LegalizeSunmmioGemm()(mod)
    if _target_utils.ENABLE_SUNMMIO_REGION_VALIDATION:
        mod = tilelang.transform.ValidateTileViewRegions()(mod)

    LayoutVisual(mod)
    mod = tilelang.transform.ValidateDynamicCommPutRoutes()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.transform.LegalizeTilesLoop()(mod)
    mod = tilelang.transform.TilesLoop()(mod)
    mod = tilelang.transform.SunmmioTileLoopFusion()(mod)

    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    mod = tilelang.transform.HoistBlockAnnotationsToFuncAttrs()(mod)
    mod = tilelang.transform.ResolveSunmmioUnit()(mod)
    return mod


def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    # Bind the target device information to the module
    """
    Bind target information and progressively legalize and lower frontend Tile IR into a form suitable for downstream optimization and codegen.

    This pass pipeline:
    - Binds the provided target to the module.
    - Legalizes frontend Tile IR into TVM-compatible constructs.
    - Simplifies expressions.
    - Configures reducer layouts and performs layout inference for fragments and shared memory.
    - Lowers high-level tile operations and L2 persistent maps.
    - Legalizes vectorized loops and inserts safety checks for memory accesses.
    - Re-simplifies to remove redundancies introduced by safety checks.
    - Attempts loop vectorization for dynamic-shaped loops.

    Parameters:
        mod (IRModule): The input IR module containing frontend Tile IR.
        target (Target): Target device information to bind into the module.

    Returns:
        IRModule: The transformed module, ready for target-specific optimization passes.
    """
    if target_is_sunmmio(target):
        return LowerAndLegalizeSunmmio(mod, target)

    mod = tir.transform.BindTarget(target)(mod)
    if should_force_let_inline():
        # Force-let inline whenever the pass config requests it.
        mod = tilelang.transform.LetInline()(mod)
    # Add wrapper for single buf store
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    # Normalize negative indices to canonical non-negative form
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    # Verify parallel loop correctness
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    # Inject assumes to speedup tvm prover
    mod = tilelang.transform.InjectAssumes()(mod)
    # Simplify the IR expressions
    mod = tilelang.transform.Simplify()(mod)
    # Infer memory layouts — target-conditional
    mod = tilelang.transform.LayoutReducer()(mod)
    # CUDA/ROCm/Metal: Fragment-based layout inference
    mod = tilelang.transform.LayoutInference()(mod)
    # Visualize the layout
    LayoutVisual(mod, target)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)
    # Plan and lower T.Tiles scopes into final tiled loops.
    mod = tilelang.transform.LowerTilesLoop()(mod)
    # Tile Loop Fusion
    mod = tilelang.transform.SunmmioTileLoopFusion()(mod)
    # Lower l2 persistent map
    mod = tilelang.transform.LowerL2Persistent()(mod)
    # Decouple type cast vectorization constraints before vectorization
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # Lower frontend pointer metadata op to standard tvm_access_ptr
    mod = tilelang.transform.LowerAccessPtr()(mod)
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    # use an enhanced pass to simplify the dynamic symbolics
    # TODO(lei): return to tir pass when kSymbolicBound simplification
    # is merged into tvm.
    mod = tilelang.transform.Simplify()(mod)
    # Hoist any root-block annotations to PrimFunc attrs if pass is available
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    return mod


_SUNMMIO_ALIAS_PROOF_ATTR = "tl.sunmmio.alias_analysis.proven"
_SUNMMIO_ALIAS_PROOF_REASON_ATTR = "tl.sunmmio.alias_analysis.reason"


def _infer_sunmmio_noalias_prim_func(func: tir.PrimFunc) -> tir.PrimFunc:
    """Promote the noalias contract already guaranteed by device extraction.

    ``SplitHostDevice`` marks every extracted SunMMIO device function with
    ``tir.noalias`` and carries ``tl.non_restrict_params`` as the explicit
    exception list. Pipeline planning runs before that extraction, so it
    otherwise cannot use the same contract. This bounded analysis checks that
    every handle parameter has one independent global Buffer root and that no
    exception was requested, then makes the later ABI fact visible early.

    This is deliberately fail-closed. It does not infer anything for a
    non-SunMMIO target, an explicit ``tir.noalias`` setting, an opaque handle,
    a shared parameter root, or a non-restrict parameter. Internal buffer
    views are handled separately by the existing residency and sync analyses;
    parameter noalias does not claim that those views are independent.
    """
    attrs = func.attrs
    if attrs is None or "target" not in attrs:
        return func
    if not target_is_sunmmio(attrs["target"]):
        return func
    if "tir.noalias" in attrs:
        return func

    if "tl.non_restrict_params" in attrs:
        non_restrict = attrs["tl.non_restrict_params"]
        if len(non_restrict) != 0:
            return func

    pointer_param_count = 0
    storage_roots = set()
    for param in func.params:
        if str(param.dtype) != "handle":
            continue
        pointer_param_count += 1
        if param not in func.buffer_map:
            return func
        buffer = func.buffer_map[param]
        if str(buffer.scope()) not in {"", "global"}:
            return func
        root = buffer.data
        if root in storage_roots:
            return func
        storage_roots.add(root)

    if pointer_param_count == 0:
        return func

    func = func.with_attr("tir.noalias", True)
    func = func.with_attr(_SUNMMIO_ALIAS_PROOF_ATTR, True)
    return func.with_attr(
        _SUNMMIO_ALIAS_PROOF_REASON_ATTR,
        "sunmmio_device_abi_distinct_parameter_roots",
    )


def _infer_sunmmio_noalias(mod: IRModule) -> IRModule:
    """Run the bounded compile-time alias proof on every PrimFunc."""

    def pass_fn(func: tir.PrimFunc, _mod: IRModule, _ctx):
        return _infer_sunmmio_noalias_prim_func(func)

    return tir.transform.prim_func_pass(pass_fn, opt_level=0, name="tl.InferSunmmioNoAlias")(mod)


def _record_sunmmio_multidim_atomic_fallback(stage: str, error: Exception) -> None:
    """Record a complete Multidim-to-1D retry without making diagnostics mandatory."""
    payload = {
        "schema_version": 1,
        "fallback": "complete_module_to_1d_ilp",
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    _LOGGER.warning(
        "Sunmmio Multidim auto mode failed at %s; retrying the original module with 1D ILP: %s",
        stage,
        error,
    )
    output_path = os.environ.get("TL_SUNMMIO_MULTIDIM_ATOMIC_FALLBACK_JSON")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _apply_sunmmio_one_dimensional_ilp(mod: IRModule) -> IRModule:
    mod = tilelang.transform.SunmmioPipelinePlanningILP(debug=False)(mod)
    return tilelang.transform.InjectSunmmioPipelineILP()(mod)


_STATIC_WORKITEM_APPLIED_ATTRS = (
    "tl.sunmmio.multidim.workitem_flatten.applied",
    "tl.sunmmio.multidim.a_residency.applied",
)
_STATIC_WORKITEM_POLICY_ENV = "TL_SUNMMIO_STATIC_WORKITEM_POLICY"


def _module_has_static_workitem_transform(mod: IRModule) -> bool:
    """Whether proof-driven static WorkItem rewriting changed any PrimFunc."""
    for base_func in mod.functions.values():
        if not isinstance(base_func, tir.PrimFunc) or base_func.attrs is None:
            continue
        for name in _STATIC_WORKITEM_APPLIED_ATTRS:
            value = base_func.attrs.get(name)
            if isinstance(value, tir.IntImm) and int(value.value) != 0:
                return True
    return False


def _get_static_workitem_candidate_policy() -> str:
    policy = os.environ.get(_STATIC_WORKITEM_POLICY_ENV, "auto")
    if policy not in {"off", "auto", "forced"}:
        raise ValueError(f"{_STATIC_WORKITEM_POLICY_ENV} must be 'off', 'auto', or 'forced', got {policy!r}")
    return policy


def _apply_sunmmio_static_workitem_candidate(original: IRModule, policy: str) -> tuple[IRModule, IRModule] | None:
    """Build the generic static-WorkItem Hierarchical ILP candidate.

    This route has no kernel, shape, or buffer-name checks. The existing
    proof-driven flatten pass is the eligibility gate. A non-applicable
    module returns None so ordinary generic Multidim remains unchanged.
    """
    pass_policy = "forced" if policy == "forced" else "auto"
    candidate = original
    candidate = tilelang.transform.FlattenSunmmioStaticWorkItems(policy=pass_policy)(candidate)
    if not _module_has_static_workitem_transform(candidate):
        return None
    planned = tilelang.transform.SunmmioPipelinePlanningILP(debug=False)(candidate)
    planned = tilelang.transform.SunmmioRegionCrossOuterPipelinePlanning(
        debug=False,
        policy=pass_policy,
        strategy="hierarchical_ilp",
    )(planned)
    selected = tilelang.transform.InjectSunmmioPipelineILP()(planned)
    return planned, selected


def _apply_sunmmio_ordinary_multidim_candidate(original: IRModule, policy: str) -> tuple[IRModule, IRModule]:
    """Build the ordinary residency/boundary Multidim candidate."""
    candidate = original
    candidate = tilelang.transform.AnalyzeSunmmioTileResidency(policy=os.environ.get("TL_SUNMMIO_RESIDENCY_POLICY", "auto_strict"))(
        candidate
    )
    planned = tilelang.transform.SunmmioMultidimPipelinePlanningILP(debug=False, policy=policy)(candidate)
    selected = tilelang.transform.InjectSunmmioMultidimPipelineILP(policy=policy)(planned)
    return planned, selected


def _apply_sunmmio_multidim_atomic(mod: IRModule, policy: str) -> IRModule:
    """Select one legal Multidim candidate or retry the untouched input."""
    original = mod
    static_workitem_policy = _get_static_workitem_candidate_policy()
    static_candidate = None
    if static_workitem_policy != "off":
        try:
            static_candidate = _apply_sunmmio_static_workitem_candidate(original, static_workitem_policy)
            if static_candidate is not None and (policy != "auto" or static_workitem_policy == "forced"):
                return static_candidate[1]
        except Exception as error:
            if static_workitem_policy == "forced":
                raise
            _LOGGER.warning(
                "Sunmmio static WorkItem candidate rejected without changing ordinary Multidim selection: %s",
                error,
            )

    try:
        # A non-selected planner must not overwrite official diagnostic files.
        if static_candidate is not None:
            with _sunmmio_shadow_planning_scope():
                ordinary_candidate = _apply_sunmmio_ordinary_multidim_candidate(original, policy)
        else:
            ordinary_candidate = _apply_sunmmio_ordinary_multidim_candidate(original, policy)
        ordinary_planned, ordinary_selected = ordinary_candidate
        if static_candidate is None:
            return ordinary_selected

        static_planned, static_selected = static_candidate
        comparison = compare_pipeline_candidate_work(static_planned, ordinary_planned)
        if comparison != 1:
            # Equal or incomparable candidates preserve the established static
            # WorkItem priority and its measured CK_MOE behavior.
            return static_selected

        # Rebuild the winner outside the shadow scope so official diagnostics
        # describe the selected candidate.
        _, ordinary_selected = _apply_sunmmio_ordinary_multidim_candidate(original, policy)
        return ordinary_selected
    except Exception as error:
        if policy != "auto":
            raise
        if static_candidate is not None:
            return static_candidate[1]
        _record_sunmmio_multidim_atomic_fallback("analysis_planning_or_injection", error)
        return _apply_sunmmio_one_dimensional_ilp(original)


def _lower_sunmmio_after_pipeline(mod: IRModule) -> IRModule:
    """Run the common lowering sequence, including dependency-driven sync."""
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tir.transform.Simplify()(mod)

    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tilelang.transform.RemoveUnusedSunmmioAllocations()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.InjectSunmmioSync()(mod)

    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    return tilelang.transform.LowerDeviceKernelLaunch()(mod)


def OptimizeForSunmmio(mod: IRModule, target: Target) -> IRModule:
    pass_ctx = tilelang.transform.get_pass_context()
    # Make the downstream SunMMIO device ABI alias contract available to
    # pipeline planning. Unknown or explicitly non-restrict parameters remain
    # unchanged and therefore keep the existing conservative fallback.
    mod = _infer_sunmmio_noalias(mod)
    mod = tilelang.transform.IfStmtBinding()(mod)
    multidim_original = None
    multidim_policy = None
    if should_enable_sunmmio_pipeline(pass_ctx):
        pipeline_mode = get_sunmmio_pipeline_mode(pass_ctx)
        if pipeline_mode == "multidim_ilp":
            # Save the exact pre-Multidim module.  TVM passes are copy-on-write,
            # so this is the transaction boundary used by both retry paths.
            multidim_original = mod
            multidim_policy = get_sunmmio_multidim_policy(pass_ctx)
            mod = _apply_sunmmio_multidim_atomic(mod, multidim_policy)
        elif pipeline_mode == "ilp":
            mod = _apply_sunmmio_one_dimensional_ilp(mod)
        elif pipeline_mode in {"hierarchical_ilp", "c_style_pipeline"}:
            region_policy = get_sunmmio_region_cross_outer_policy(pass_ctx)
            if pipeline_mode == "c_style_pipeline":
                mod = tilelang.transform.FlattenSunmmioStaticWorkItems(policy=region_policy)(mod)
            mod = tilelang.transform.SunmmioPipelinePlanningILP(debug=False)(mod)
            mod = tilelang.transform.SunmmioRegionCrossOuterPipelinePlanning(
                debug=False,
                policy=region_policy,
                strategy="hierarchical_ilp",
            )(mod)
            mod = tilelang.transform.InjectSunmmioPipelineILP()(mod)
        elif pipeline_mode == "cross_outer":
            cross_outer_policy = get_sunmmio_cross_outer_policy(pass_ctx)
            mod = tilelang.transform.SunmmioCrossOuterPipelinePlanning(debug=False, policy=cross_outer_policy)(mod)
            mod = tilelang.transform.InjectSunmmioCrossOuterPipeline(policy=cross_outer_policy)(mod)
            if cross_outer_policy == "auto":
                mod = tilelang.transform.SunmmioPipelinePlanning(debug=False)(mod)
                mod = tilelang.transform.InjectSunmmioPipeline()(mod)
        elif pipeline_mode == "region_cross_outer":
            region_policy = get_sunmmio_region_cross_outer_policy(pass_ctx)
            region_strategy = get_sunmmio_region_cross_outer_strategy(pass_ctx)
            mod = tilelang.transform.SunmmioRegionCrossOuterPipelinePlanning(
                debug=False,
                policy=region_policy,
                strategy=region_strategy,
            )(mod)
            mod = tilelang.transform.InjectSunmmioRegionCrossOuterPipeline(
                policy=region_policy,
                strategy=region_strategy,
            )(mod)
            if region_policy == "auto":
                mod = tilelang.transform.SunmmioPipelinePlanning(debug=False)(mod)
                mod = tilelang.transform.InjectSunmmioPipeline()(mod)
        else:
            mod = tilelang.transform.SunmmioPipelinePlanning(debug=False)(mod)
            mod = tilelang.transform.InjectSunmmioPipeline()(mod)

    try:
        return _lower_sunmmio_after_pipeline(mod)
    except Exception as error:
        # Sync placement and later lowering are part of the Multidim
        # transaction.  Retry once from the original module, never from a
        # partially rewritten Multidim module.
        if multidim_policy != "auto" or multidim_original is None:
            raise
        _record_sunmmio_multidim_atomic_fallback("post_pipeline_or_sync", error)
        fallback = _apply_sunmmio_one_dimensional_ilp(multidim_original)
        return _lower_sunmmio_after_pipeline(fallback)


def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    if target_is_sunmmio(target):
        return OptimizeForSunmmio(mod, target)

    pass_ctx = tilelang.transform.get_pass_context()
    # Lower the shared.barrier into specific initialization slot
    mod = tilelang.transform.LowerSharedBarrier()(mod)
    # Lower the shared.tmem into specific initialization slot
    mod = tilelang.transform.LowerSharedTmem()(mod)
    # which may be introduced by the LegalizeSafeMemoryAccess
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tilelang.transform.MultiVersionBuffer()(mod)
        mod = tilelang.transform.WarpSpecialized()(mod)
        mod = tilelang.transform.InjectTmaBarrier()(mod)
        # if tma is not enabled, we can also do pipeline planning
        # to get better performance with async copy
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        # warp_specialized pass will pack the if stmt into the block
        # so we need to lower the opaque block first
        mod = tilelang.transform.LowerOpaqueBlock()(mod)
        if is_hopper(target):
            mod = tilelang.transform.RewriteWgmmaSync()(mod)
    else:
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)

    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    # ConfigIndexBitwidth must be applied after FlattenBuffer
    # as it will flatten index computing
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl only use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)
    mod = tilelang.transform.LowerLDGSTG()(mod)
    mod = tilelang.transform.LowerHopperIntrin()(mod)
    # Global Barrier Synchronization must be applied before
    # SplitHostDevice pass, as the global barrier
    if allow_global_thread_synchronization():
        mod = tilelang.transform.ThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    # Mark the function contains pdl_sync or pdl_trigger
    mod = tilelang.transform.MarkCudaSyncCalls(have_pdl(target))(mod)

    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    # MergeSharedMemoryAllocations must be applied after SplitHostDevice
    # because the merged allocation site is at the beginning of each device function
    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge)(mod)
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.InjectFenceProxy()(mod)
    else:
        if allow_fence_proxy(target=target):
            # in hopper device, wgmma is an async proxy
            # so we need to inject a fence proxy before it
            mod = tilelang.transform.InjectFenceProxy()(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    # Inject PTX async copy must behind the thread sync pass
    # as ptx async copy won't be recognized as a valid buffer load
    mod = tilelang.transform.InjectPTXAsyncCopy()(mod)
    if allow_tma_and_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.AnnotateWarpGroupRegAlloc()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)

    # Transform threadblock to persistent threadblock
    mod = tilelang.transform.PersistThreadblock()(mod)

    return mod
