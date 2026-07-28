import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


tilelang.env.disable_cache()


def missing_span_kernel():
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    func = tvm.tir.PrimFunc([], body).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": func})


def missing_body_span_kernel_with_function_span():
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    span = tvm.ir.Span(
        tvm.ir.SourceName("test_check_span.py"),
        7,
        7,
        3,
        20,
    )
    func = tvm.tir.PrimFunc([], body, span=span).with_attr(
        "global_symbol",
        "main",
    )
    return tvm.IRModule({"main": func})


def test_check_span_defaults_to_warning(capfd):
    mod = tilelang.transform.CheckSpan()(missing_span_kernel())

    message = capfd.readouterr().err
    assert isinstance(mod["main"], tvm.tir.PrimFunc)
    assert "CheckSpan found" in message
    assert "TIR node(s) without Span" in message


def test_check_span_warning_reports_and_continues(capfd):
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_CHECK_SPAN_LOG_LEVEL: "WARNING"}):
        mod = tilelang.transform.CheckSpan()(missing_span_kernel())

    message = capfd.readouterr().err
    assert isinstance(mod["main"], tvm.tir.PrimFunc)
    assert "CheckSpan found" in message
    assert "TIR node(s) without Span" in message
    assert "tir.PrimFunc" in message
    assert "tir.Evaluate" in message
    assert "ir.IntImm" in message


def test_check_span_warning_appends_available_dsl_span(capfd):
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_CHECK_SPAN_LOG_LEVEL: "WARNING"}):
        tilelang.transform.CheckSpan()(missing_body_span_kernel_with_function_span())

    message = capfd.readouterr().err
    assert "CheckSpan found" in message
    assert "at TileLang DSL: test_check_span.py:7:3" in message


def test_check_span_fatal_reports_and_stops():
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_CHECK_SPAN_LOG_LEVEL: "FATAL"}), pytest.raises(Exception) as exc_info:
        tilelang.transform.CheckSpan()(missing_span_kernel())

    message = str(exc_info.value)
    assert "CheckSpan found" in message
    assert "TIR node(s) without Span" in message
    assert "tir.PrimFunc" in message
    assert "tir.Evaluate" in message
    assert "ir.IntImm" in message


def test_check_span_rejects_invalid_log_level():
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_CHECK_SPAN_LOG_LEVEL: "ERROR"}), pytest.raises(Exception) as exc_info:
        tilelang.transform.CheckSpan()(missing_span_kernel())

    message = str(exc_info.value)
    assert "Invalid tl.check_span_log_level value `ERROR`" in message
    assert "expected `FATAL` or `WARNING`" in message


@tilelang.jit(
    target="sunmmio",
    execution_backend="sunmmio_sunsim",
    pass_configs={tilelang.PassConfigKey.TL_CHECK_SPAN_LOG_LEVEL: "WARNING"},
)
def check_span_warning_jit_kernel():
    @T.prim_func
    def main(
        source: T.Tensor((32, 32), "float32"),
        output: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel():
            source_shared = T.alloc_shared(
                (32, 32),
                "float32",
                scope="shared.rsram",
            )
            output_shared = T.alloc_shared(
                (32, 32),
                "float32",
                scope="shared.rsram",
            )
            T.copy(source[0, 0], source_shared)
            for i, j in T.Tiles([32, 32]):
                output_shared[i, j] = source_shared[i, j] + 1.0
            T.copy(output_shared, output[0, 0])

    return main


def test_check_span_warning_runs_in_jit_pipeline(capfd):
    kernel = check_span_warning_jit_kernel()

    message = capfd.readouterr().err
    assert kernel is not None
    assert "CheckSpan found" in message
    assert "after target lowering" in message


@tilelang.jit(
    target="sunmmio",
    execution_backend="sunmmio_sunsim",
    pass_configs={tilelang.PassConfigKey.TL_CHECK_SPAN_LOG_LEVEL: "FATAL"},
)
def check_span_fatal_jit_kernel():
    @T.prim_func
    def main(
        source: T.Tensor((32, 32), "float32"),
        output: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel():
            source_shared = T.alloc_shared(
                (32, 32),
                "float32",
                scope="shared.rsram",
            )
            output_shared = T.alloc_shared(
                (32, 32),
                "float32",
                scope="shared.rsram",
            )
            T.copy(source[0, 0], source_shared)
            for i, j in T.Tiles([32, 32]):
                output_shared[i, j] = source_shared[i, j] + 1.0
            T.copy(output_shared, output[0, 0])

    return main


def test_check_span_fatal_stops_jit_before_codegen():
    with pytest.raises(Exception) as exc_info:
        check_span_fatal_jit_kernel()

    message = str(exc_info.value)
    print(message)
    assert "CheckSpan found" in message
    assert "after target lowering" in message


if __name__ == "__main__":
    tilelang.testing.main()
