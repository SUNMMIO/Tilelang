from functools import wraps
import os
import re
import warnings
import tilelang
from tilelang import tvm
from tilelang.transform import PassConfigKey
from tilelang.utils.target import determine_target
from typing import Any, Literal, Callable
from tvm.target import Target
from tilelang.language.eager import PrimFunc
from tvm import tir, IRModule
from tvm.ir import CallingConv
from tilelang.engine.param import KernelParam
from tilelang.transform import PassContext
from tilelang.contrib.nvcc import have_tma
from tilelang.utils.target import target_is_sunmmio
from tilelang.jit.adapter.utils import is_cuda_target


def target(target_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tvm.target.Target(determine_target(target_name, return_object=True)):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def allow_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
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


def LayoutVisual(mod: IRModule) -> None:
    if should_enable_layout_visual():
        formats = get_layout_visual_formats()
        tilelang.analysis.LayoutVisual(formats=formats)(mod)


def is_cpu_device_backend(target: Target):
    return target.kind.name == "c"


def has_device_kernel_launch(attrs) -> bool:
    """Check if the attributes indicate a device kernel launch."""
    return bool(attrs and "calling_conv" in attrs and attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH)


def is_device_call_c_device(func: tir.PrimFunc):
    attrs = func.attrs
    calling_conv = attrs.get("calling_conv", CallingConv.DEFAULT)
    is_cpacked = calling_conv == CallingConv.C_PACKED_FUNC

    # Check if it's a C target
    if "target" in attrs and attrs["target"].kind.name == "c" and not is_cpacked:
        return True

    return has_device_kernel_launch(attrs)


def is_device_call(func: tir.PrimFunc):
    return has_device_kernel_launch(func.attrs)


def get_device_call(is_device_c: bool = False) -> Callable[[tir.PrimFunc], bool]:
    return is_device_call_c_device if is_device_c else is_device_call


def get_host_call(is_device_c: bool = False) -> Callable[[tir.PrimFunc], bool]:
    return lambda func: not get_device_call(is_device_c)(func)


def is_sunmmio_call(func: tir.PrimFunc):
    attrs = func.attrs
    return bool(attrs and "target" in attrs and target_is_sunmmio(attrs["target"]))


def get_device_call_sunmmio() -> Callable[[tir.PrimFunc], bool]:
    return is_sunmmio_call


def get_host_call_sunmmio() -> Callable[[tir.PrimFunc], bool]:
    return lambda func: not is_sunmmio_call(func)


def extrac_params(func: tir.PrimFunc) -> list[KernelParam]:
    tensor_types = []
    for var in func.params:
        if var in func.buffer_map:
            tensor_types.append(KernelParam.from_buffer(func.buffer_map[var]))
        else:
            tensor_types.append(KernelParam.from_var(var))
    return tensor_types


def canon_target_host(target: str | Target, target_host: str | Target | None):
    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "c"
    return target_host


def PreLowerSemanticCheck(mod: IRModule) -> None:
    if should_enable_ast_print():
        tilelang.analysis.ASTPrinter()(mod)
    tilelang.analysis.NestedLoopChecker()(mod)
    tilelang.analysis.FragmentLoopChecker()(mod)


def pass_test(mod: IRModule, pass_name: str, test_config: dict[str, Any]) -> None:
    if pass_name in test_config:
        test_info = test_config[pass_name]
        print(f"testing {pass_name}")
        if "script_expected" in test_info:
            expect = test_info["script_expected"]
            if isinstance(expect, str):
                expect = [expect.strip()]
            elif isinstance(expect, list):
                for lines in expect:
                    assert isinstance(lines, str), f"Invalid type for script_expected: {type(lines)}"
                expect = [lines.strip() for lines in expect]
            else:
                raise ValueError(f"Invalid type for script_expected: {type(expect)}")
            script = mod.script(show_meta=True).strip()
            error_msg = f"The generated script of {pass_name} does not match the expected output."
            if "show_generated_script" in test_info and test_info["show_generated_script"]:
                error_msg = error_msg + f"\nGenerated script:\n{script}"
            for lines in expect:
                if lines not in script:
                    warnings.warn(error_msg, stacklevel=2)

        if "formal_verify" in test_info:
            formal_verify = test_info["formal_verify"]
            if isinstance(formal_verify, list):
                for test_func in formal_verify:
                    test_func(mod)
            else:
                formal_verify(mod)


def LowerAndLegalize_sunmmio_test(
    mod: IRModule,
    target: Target,
    test_config: dict[str, Any] | None = None,
    log_pass_output: bool = False,
    show_meta: bool = False,
    log_dir: str = "./",
    log_passes: list[str] | None = None,
) -> IRModule:
    if test_config is None:
        test_config = {}
    out_file = os.path.join(log_dir, "passes_lower_and_legalize.log")
    if log_pass_output:
        print(f"Logging pass output in LowerAndLegalize to {out_file}")
        with open(out_file, "w") as f:
            f.write("\n'=== Initial Mod ==='\n")
            if log_passes is None:
                f.write(mod.script(show_meta=show_meta).strip() + "\n\n")

    def pass_output_process(mod: IRModule, pass_name: str, test_config: dict[str, Any]) -> None:
        if log_pass_output and (log_passes is None or pass_name in log_passes):
            with open(out_file, "a") as f:
                f.write(f"'=== After Pass {pass_name} ==='\n")
                f.write(mod.script(show_meta=show_meta).strip() + "\n\n")
        pass_test(mod, pass_name, test_config)

    mod = tir.transform.BindTarget(target)(mod)
    pass_output_process(mod, "BindTarget", test_config)

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
        pass_output_process(mod, "LetInline", test_config)

    # mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    pass_output_process(mod, "LegalizeNegativeIndex", test_config)

    # if should_enable_race_check():
    #     mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    pass_output_process(mod, "InjectAssumes", test_config)

    mod = tilelang.transform.Simplify()(mod)
    pass_output_process(mod, "Simplify_lower_1", test_config)

    mod = tilelang.transform.InferSramScope()(mod)
    pass_output_process(mod, "InferSramScope", test_config)

    mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
    pass_output_process(mod, "LegalizeSunmmioDataPath", test_config)

    # mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.SunmmioLayoutInference()(mod)
    pass_output_process(mod, "SunmmioLayoutInference", test_config)

    mod = tilelang.transform.LegalizeSunmmioGemm()(mod)
    pass_output_process(mod, "LegalizeSunmmioGemm", test_config)

    LayoutVisual(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    pass_output_process(mod, "LowerTileOp", test_config)

    mod = tilelang.transform.LegalizeTilesLoop()(mod)
    pass_output_process(mod, "LegalizeTilesLoop", test_config)

    mod = tilelang.transform.TilesLoop()(mod)
    pass_output_process(mod, "TilesLoop", test_config)

    mod = tilelang.transform.SunmmioTileLoopFusion()(mod)
    pass_output_process(mod, "SunmmioTileLoopFusion", test_config)
    # mod = tilelang.transform.LowerL2Persistent()(mod)
    # mod = tilelang.transform.DecoupleTypeCast()(mod)
    # pass_output_process(mod, "DecoupleTypeCast", test_config)

    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    pass_output_process(mod, "LegalizeVectorizedLoop", test_config)

    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    pass_output_process(mod, "LegalizeSafeMemoryAccess", test_config)

    mod = tilelang.transform.LowerAccessPtr()(mod)
    pass_output_process(mod, "LowerAccessPtr", test_config)

    mod = tilelang.transform.Simplify()(mod)
    pass_output_process(mod, "Simplify_lower_2", test_config)

    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    pass_output_process(mod, "HoistNonRestrictParams", test_config)

    mod = tilelang.transform.HoistBlockAnnotationsToFuncAttrs()(mod)
    pass_output_process(mod, "HoistBlockAnnotationsToFuncAttrs", test_config)

    return mod


def OptimizeForSunmmio_test(
    mod: IRModule,
    target: Target,
    test_config: dict[str, Any] | None = None,
    log_pass_output: bool = False,
    show_meta: bool = False,
    log_dir: str = "./",
    log_passes: list[str] | None = None,
) -> IRModule:
    if test_config is None:
        test_config = {}
    out_file = os.path.join(log_dir, "passes_optimize_for_sunmmio.log")
    if log_pass_output:
        print(f"Logging pass output in OptimizeForSunmmio to {out_file}")
        with open(out_file, "w") as f:
            f.write("\n'=== Initial Mod ==='\n")
            if log_passes is None:
                f.write(mod.script(show_meta=show_meta).strip() + "\n\n")

    def pass_output_process(mod: IRModule, pass_name: str, test_config: dict[str, Any]) -> None:
        if log_pass_output and (log_passes is None or pass_name in log_passes):
            with open(out_file, "a") as f:
                f.write(f"'=== After Pass {pass_name} ==='\n")
                f.write(mod.script(show_meta=show_meta).strip() + "\n\n")
        pass_test(mod, pass_name, test_config)

    mod = tilelang.transform.IfStmtBinding()(mod)
    pass_output_process(mod, "IfStmtBinding", test_config)

    mod = tilelang.transform.SunmmioPipelinePlanning(debug=False)(mod)
    pass_output_process(mod, "SunmmioPipelinePlanning", test_config)

    mod = tilelang.transform.InjectSunmmioPipeline()(mod)
    pass_output_process(mod, "InjectSunmmioPipeline", test_config)

    # mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    # pass_output_process(mod, "PlanAndUpdateBufferAllocationLocation", test_config)

    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    pass_output_process(mod, "LowerOpaqueBlock", test_config)

    mod = tilelang.transform.Simplify()(mod)
    pass_output_process(mod, "Simplify_optimize_1", test_config)

    mod = tir.transform.NarrowDataType(32)(mod)
    pass_output_process(mod, "NarrowDataType", test_config)

    # mod = tilelang.transform.FlattenBuffer()(mod)
    # pass_output_process(mod, "FlattenBuffer", test_config)

    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    pass_output_process(mod, "ConfigIndexBitwidth", test_config)

    mod = tir.transform.Simplify()(mod)
    pass_output_process(mod, "Simplify_optimize_2", test_config)

    # mod = tilelang.transform.VectorizeLoop(enable_vectorize=True)(mod)
    # pass_output_process(mod, "VectorizeLoop", test_config)

    # mod = tilelang.transform.StorageRewrite()(mod)
    # pass_output_process(mod, "StorageRewrite", test_config)

    mod = tilelang.transform.LoopUnswitching()(mod)
    pass_output_process(mod, "LoopUnswitching", test_config)

    mod = tir.transform.UnrollLoop()(mod)
    pass_output_process(mod, "UnrollLoop", test_config)

    mod = tir.transform.RenormalizeSplitPattern()(mod)
    pass_output_process(mod, "RenormalizeSplitPattern", test_config)

    mod = tir.transform.Simplify()(mod)
    pass_output_process(mod, "Simplify_optimize_3", test_config)

    # return
    mod = tir.transform.RemoveNoOp()(mod)
    pass_output_process(mod, "RemoveNoOp", test_config)

    mod = tir.transform.HoistIfThenElse()(mod)
    pass_output_process(mod, "HoistIfThenElse", test_config)

    mod = tir.transform.VerifyMemory()(mod)
    pass_output_process(mod, "VerifyMemory", test_config)

    mod = tir.transform.AnnotateEntryFunc()(mod)
    pass_output_process(mod, "AnnotateEntryFunc", test_config)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    pass_output_process(mod, "AnnotateDeviceRegions", test_config)

    mod = tilelang.transform.SplitHostDevice()(mod)
    pass_output_process(mod, "SplitHostDevice", test_config)

    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    pass_output_process(mod, "AnnotateReadOnlyParams", test_config)

    mod = tilelang.transform.MergeIfStmt()(mod)
    pass_output_process(mod, "MergeIfStmt", test_config)

    mod = tilelang.transform.InjectSunmmioSync()(mod)
    pass_output_process(mod, "InjectSunmmioSync", test_config)

    mod = tilelang.transform.MakePackedAPI()(mod)
    pass_output_process(mod, "MakePackedAPI", test_config)

    mod = tilelang.transform.Simplify()(mod)
    pass_output_process(mod, "Simplify_optimize_4", test_config)

    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    pass_output_process(mod, "LowerDeviceKernelLaunch", test_config)

    return mod


def process_passes_output(log_dir, filenames, remove_header=False):
    def clean_header(text):
        if not remove_header:
            return text
        # Header to remove:
        # # from tvm.script import ir as I
        # # from tvm.script import tir as T
        #
        # @I.ir_module
        header_pattern = r"# from tvm\.script import ir as I\s*\n# from tvm\.script import tir as T\s*\n\s*\n@I\.ir_module\s*\n"
        text = re.sub(header_pattern, "", text)

        # Metadata omitted comment to remove
        meta_pattern = r"\n\s*\n# Metadata omitted\. Use show_meta=True in script\(\) method to show it\.\s*\n\n"
        text = re.sub(meta_pattern, "\n", text)
        return text

    for filename in filenames:
        log_file = os.path.join(log_dir, filename)
        if not os.path.exists(log_file):
            continue

        with open(log_file, "r") as f:
            content = f.read()

        # Split by the "=== After Pass ... ===" or "=== Initial Mod ===" markers
        # We need to capture the markers to keep them
        pattern = r"('(?:=== Initial Mod ===|=== After Pass [^=]+ ===)')"
        parts = re.split(pattern, content)

        if len(parts) < 3:
            continue

        new_parts = [parts[0]]
        # Always keep the first pass (Initial Mod)
        new_parts.append(parts[1])
        new_parts.append("\n" + clean_header(parts[2]).strip() + "\n")

        last_content = parts[2].strip()

        for i in range(3, len(parts), 2):
            header = parts[i]
            current_content = parts[i + 1]

            # Compare current content with last content
            if current_content.strip() == last_content:
                new_parts.append(header)
                new_parts.append("\nNo change.\n")
            else:
                new_parts.append(header)
                new_parts.append("\n" + clean_header(current_content).strip() + "\n")
                last_content = current_content.strip()

        with open(log_file, "w") as f:
            f.write("".join(new_parts))


def compile_test(
    func: PrimFunc = None,
    out_idx: list[int] | int | None = None,
    execution_backend: (Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None) = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
    test_config: dict[str, Any] | None = None,
    log_pass_output: bool = False,
    show_meta: bool = False,
    log_dir: str = "./",
    remove_header: bool = False,
    log_passes: list[str] | None = None,
):
    """
    Compile the given TileLang PrimFunc with TVM and return the host_mod and device_mod.
    This function mimics tilelang.jit.compile but exposes the intermediate TIR modules.
    It manually implements the lower logic to avoid failing at codegen step.

    Returns:
        tuple: (host_mod, device_mod) corresponding to the modules generated in lower.log
    """
    if test_config is None:
        test_config = {}

    for pass_name in test_config:
        for key in test_config[pass_name]:
            assert key in [
                "script_expected",
                "show_generated_script",
                "formal_verify",
            ], f"wrong key :{key} for pass {pass_name}"

    if execution_backend is None:
        execution_backend = "tvm_ffi"

    if execution_backend == "auto":
        execution_backend = "tvm_ffi"

    if pass_configs is None:
        pass_configs = {}

    if compile_flags is not None:
        compile_flags_cfg = pass_configs.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS)
        pass_configs[PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = (
            compile_flags_cfg + compile_flags if compile_flags_cfg is not None else compile_flags
        )

    # Determine target
    target = determine_target(target, return_object=True)

    # Custom Lower implementation
    func_or_mod = func

    mod = func_or_mod
    if isinstance(func_or_mod, tir.PrimFunc):
        func = func_or_mod
        mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    if isinstance(target, str):
        target = determine_target(target)

    target_host = canon_target_host(target, target_host)

    target_host = tvm.target.Target.canon_target(target_host)
    target = tvm.target.Target(target, target_host)

    if target_is_sunmmio(target):
        _is_host_call = get_host_call_sunmmio()
        _is_device_call = get_device_call_sunmmio()
    else:
        _is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
        _is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))

    with tvm.transform.PassContext(opt_level=3, config=pass_configs), target:
        # Before lowering, do semantic check
        PreLowerSemanticCheck(mod)

        if log_pass_output:
            os.makedirs(log_dir, exist_ok=True)

        # Phase 1: Lower and legalize the IR module
        mod = LowerAndLegalize_sunmmio_test(mod, target, test_config, log_pass_output, show_meta, log_dir, log_passes)

        # Phase 2: Optimize the IR for the target
        mod = OptimizeForSunmmio_test(mod, target, test_config, log_pass_output, show_meta, log_dir, log_passes)
        host_mod = tir.transform.Filter(_is_host_call)(mod)
        device_mod = tir.transform.Filter(_is_device_call)(mod)

        out_file = os.path.join(log_dir, "passes_optimize_for_sunmmio.log")

        def _log_pass(pass_name, m):
            if log_pass_output and (log_passes is None or pass_name in log_passes):
                with open(out_file, "a") as f:
                    f.write(f"'=== After Pass {pass_name} ==='\n")
                    f.write(m.script(show_meta=show_meta).strip() + "\n\n")

        _log_pass("HostMod", host_mod)
        pass_test(host_mod, "HostMod", test_config)

        _log_pass("DeviceMod", device_mod)
        pass_test(device_mod, "DeviceMod", test_config)

        if log_pass_output:
            process_passes_output(
                log_dir,
                ["passes_lower_and_legalize.log", "passes_optimize_for_sunmmio.log"],
                remove_header=remove_header,
            )

        return host_mod, device_mod
