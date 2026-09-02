"""Debug helpers for lowering a TileLang DSL program to SunMMIO device TIR."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import sys
from typing import Iterable, Mapping, TextIO

from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget, PreLowerSemanticCheck
from tilelang.utils.target import determine_target, target_is_sunmmio
from tvm import tir


@dataclass(frozen=True)
class PassExecution:
    """One pass invocation observed in the production lowering pipeline."""

    phase: str
    name: str
    occurrence: int

    @property
    def selector(self) -> str:
        return f"{self.name}#{self.occurrence}"


@dataclass(frozen=True)
class PassSnapshot:
    """IRModule captured immediately before or after one pass invocation."""

    execution: PassExecution
    mod: tvm.IRModule
    when: str


@dataclass(frozen=True)
class _PassSelector:
    name: str
    occurrence: int | None


def _parse_pass_selector(selector: str) -> _PassSelector:
    if not isinstance(selector, str) or not selector:
        raise ValueError(f"Pass selector must be a non-empty string, got {selector!r}.")

    name, marker, suffix = selector.rpartition("#")
    if not marker:
        return _PassSelector(selector, None)
    if not name or not suffix.isdigit() or int(suffix) < 1:
        raise ValueError(f"Invalid pass selector {selector!r}; use 'pass.name' or 'pass.name#N' with N >= 1.")
    return _PassSelector(name, int(suffix))


def _normalize_pass_selectors(selectors: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(selectors, str):
        return (selectors,)
    return tuple(selectors)


@tvm.instrument.pass_instrument
class _PassSnapshotInstrument:
    def __init__(
        self,
        capture_before_pass_names: Iterable[str],
        capture_after_pass_names: Iterable[str],
    ):
        self.capture_before_pass_names = frozenset(capture_before_pass_names)
        self.capture_after_pass_names = frozenset(capture_after_pass_names)
        self.phase = "PreLowerSemanticCheck"
        self.executions: list[PassExecution] = []
        self.before_snapshots: list[PassSnapshot] = []
        self.after_snapshots: list[PassSnapshot] = []
        self._occurrences: dict[str, int] = defaultdict(int)

    def run_before_pass(self, mod, pass_info):
        name = pass_info.name
        if name in self.capture_before_pass_names:
            execution = PassExecution(self.phase, name, self._occurrences[name] + 1)
            self.before_snapshots.append(PassSnapshot(execution, mod, "before"))

    def run_after_pass(self, mod, pass_info):
        name = pass_info.name
        self._occurrences[name] += 1
        execution = PassExecution(self.phase, name, self._occurrences[name])
        self.executions.append(execution)
        if name in self.capture_after_pass_names:
            self.after_snapshots.append(PassSnapshot(execution, mod, "after"))


@dataclass(frozen=True)
class LoweringResult:
    """Final modules and optional per-pass snapshots from one lowering run."""

    target: tvm.target.Target
    full_mod: tvm.IRModule
    device_mod: tvm.IRModule
    executions: tuple[PassExecution, ...]
    before_snapshots: tuple[PassSnapshot, ...]
    after_snapshots: tuple[PassSnapshot, ...]

    @property
    def snapshots(self) -> tuple[PassSnapshot, ...]:
        """Backward-compatible alias for snapshots captured after passes."""

        return self.after_snapshots

    def select_pass_snapshots(
        self,
        selector: str,
        *,
        when: str = "after",
    ) -> tuple[PassSnapshot, ...]:
        """Return before/after snapshots matching ``pass.name`` or ``pass.name#N``."""

        if when not in ("before", "after"):
            raise ValueError(f"when must be 'before' or 'after', got {when!r}.")
        parsed = _parse_pass_selector(selector)
        source = self.before_snapshots if when == "before" else self.after_snapshots
        snapshots = tuple(
            snapshot
            for snapshot in source
            if snapshot.execution.name == parsed.name and (parsed.occurrence is None or snapshot.execution.occurrence == parsed.occurrence)
        )
        if snapshots:
            return snapshots

        executions = [execution for execution in self.executions if execution.name == parsed.name]
        if executions:
            captured = [snapshot.execution.selector for snapshot in source]
            raise ValueError(
                f"Pass {selector!r} was executed but its requested {when} result was not captured. "
                f"Captured {when} pass results: {captured}."
            )

        available = list(dict.fromkeys(execution.name for execution in self.executions))
        raise ValueError(f"Pass {parsed.name!r} was not executed. Available passes: {available}.")

    def pass_snapshot(self, selector: str, *, when: str = "after") -> PassSnapshot:
        """Return one before/after snapshot; an unnumbered repeated pass resolves to its last run."""

        return self.select_pass_snapshots(selector, when=when)[-1]

    def print_pass_pipeline(self, *, file: TextIO | None = None) -> None:
        """Print the observed production pass order and phase for discovery."""

        stream = file or sys.stdout
        for index, execution in enumerate(self.executions, start=1):
            print(
                f"{index:02d} [{execution.phase}] {execution.selector}",
                file=stream,
            )

    def print_pass_tir(
        self,
        selector: str,
        *,
        when: str = "after",
        show_meta: bool = False,
        file: TextIO | None = None,
    ) -> None:
        """Print TIR captured before or after the selected pass invocation(s)."""

        stream = file or sys.stdout
        for snapshot in self.select_pass_snapshots(selector, when=when):
            execution = snapshot.execution
            _print_ir_module(
                snapshot.mod,
                label=f"{when.title()} {execution.phase}: {execution.selector}",
                show_meta=show_meta,
                file=stream,
            )

    def print_device_tir(
        self,
        *,
        show_meta: bool = False,
        file: TextIO | None = None,
    ) -> None:
        """Print the final module containing only SunMMIO device PrimFuncs."""

        _print_ir_module(
            self.device_mod,
            label="Final SunMMIO Device TIR",
            show_meta=show_meta,
            file=file or sys.stdout,
        )


def _print_ir_module(
    mod: tvm.IRModule,
    *,
    label: str,
    show_meta: bool,
    file: TextIO,
) -> None:
    print(f"===== {label} =====", file=file)
    if file is sys.stdout:
        mod.show(show_meta=show_meta)
    else:
        print(mod.script(show_meta=show_meta), file=file)


def _make_sunmmio_target(target: tvm.target.Target | None) -> tvm.target.Target:
    if target is not None:
        return target
    device_target = determine_target("sunmmio", return_object=True)
    host_target = tvm.target.Target.canon_target("llvm")
    return tvm.target.Target(device_target, host_target)


def _as_ir_module(func_or_mod: tir.PrimFunc | tvm.IRModule) -> tvm.IRModule:
    if isinstance(func_or_mod, tvm.IRModule):
        return func_or_mod
    if isinstance(func_or_mod, tir.PrimFunc):
        if func_or_mod.attrs is None or "global_symbol" not in func_or_mod.attrs:
            raise ValueError("The TileLang PrimFunc must have a global_symbol attribute.")
        return tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
    raise TypeError(f"Expected a PrimFunc or IRModule, got {type(func_or_mod).__name__}.")


def _is_sunmmio_device_func(candidate) -> bool:
    return (
        isinstance(candidate, tir.PrimFunc)
        and candidate.attrs is not None
        and "target" in candidate.attrs
        and target_is_sunmmio(candidate.attrs["target"])
    )


def lower_to_device_tir(
    func_or_mod: tir.PrimFunc | tvm.IRModule,
    *,
    capture_passes: str | Iterable[str] = (),
    capture_before_passes: str | Iterable[str] = (),
    target: tvm.target.Target | None = None,
    pass_configs: Mapping[str, object] | None = None,
) -> LoweringResult:
    """Run the production lowering pipeline and stop before device codegen.

    ``capture_passes`` accepts exact TVM pass names. Add ``#N`` to select a
    repeated pass invocation, for example ``tl.Simplify#2``. All invocations
    of the base pass are captured so the result can select and print them.
    """

    after_selectors = _normalize_pass_selectors(capture_passes)
    before_selectors = _normalize_pass_selectors(capture_before_passes)
    capture_after_pass_names = {_parse_pass_selector(selector).name for selector in after_selectors}
    capture_before_pass_names = {_parse_pass_selector(selector).name for selector in before_selectors}
    instrument = _PassSnapshotInstrument(
        capture_before_pass_names,
        capture_after_pass_names,
    )
    target = _make_sunmmio_target(target)
    mod = _as_ir_module(func_or_mod)

    with (
        tvm.transform.PassContext(
            opt_level=3,
            config=dict(pass_configs or {}),
            instruments=[instrument],
        ),
        target,
    ):
        instrument.phase = "PreLowerSemanticCheck"
        PreLowerSemanticCheck(mod)

        instrument.phase = "LowerAndLegalize"
        mod = LowerAndLegalize(mod, target)

        instrument.phase = "OptimizeForTarget"
        mod = OptimizeForTarget(mod, target)

    device_mod = tir.transform.Filter(_is_sunmmio_device_func)(mod)
    if not device_mod.get_global_vars():
        raise RuntimeError("The lowering pipeline did not produce a SunMMIO device PrimFunc.")

    return LoweringResult(
        target=target,
        full_mod=mod,
        device_mod=device_mod,
        executions=tuple(instrument.executions),
        before_snapshots=tuple(instrument.before_snapshots),
        after_snapshots=tuple(instrument.after_snapshots),
    )


def lower_and_print_device_tir(
    func_or_mod: tir.PrimFunc | tvm.IRModule,
    *,
    print_before_passes: str | Iterable[str] = (),
    print_after_passes: str | Iterable[str] = (),
    print_final: bool = True,
    print_pass_pipeline: bool = False,
    show_meta: bool = False,
    file: TextIO | None = None,
    target: tvm.target.Target | None = None,
    pass_configs: Mapping[str, object] | None = None,
) -> LoweringResult:
    """Lower DSL and print requested intermediate pass results and device TIR.

    Examples
    --------
    ``lower_and_print_device_tir(kernel)`` prints the final device TIR.

    ``lower_and_print_device_tir(kernel, print_after_passes=("tl.LowerTileOp",
    "tl.InjectSunmmioSync"))`` also prints those intermediate modules.
    """

    before_selectors = _normalize_pass_selectors(print_before_passes)
    after_selectors = _normalize_pass_selectors(print_after_passes)
    result = lower_to_device_tir(
        func_or_mod,
        capture_passes=after_selectors,
        capture_before_passes=before_selectors,
        target=target,
        pass_configs=pass_configs,
    )

    stream = file or sys.stdout
    if print_pass_pipeline:
        result.print_pass_pipeline(file=stream)
    for selector in before_selectors:
        result.print_pass_tir(selector, when="before", show_meta=show_meta, file=stream)
    for selector in after_selectors:
        result.print_pass_tir(selector, show_meta=show_meta, file=stream)
    if print_final:
        result.print_device_tir(show_meta=show_meta, file=stream)
    return result
