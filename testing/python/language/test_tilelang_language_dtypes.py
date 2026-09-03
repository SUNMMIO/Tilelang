import pytest

from tilelang.language import dtypes
from tvm.target import datatype as tvm_datatype


def _patch_custom_dtype_registry(monkeypatch, entries):
    registered = []

    monkeypatch.setattr(tvm_datatype, "get_type_registered", lambda code: code in entries)
    monkeypatch.setattr(tvm_datatype, "get_type_name", lambda code: entries[code])

    def register(type_name, type_code):
        entries[type_code] = type_name
        registered.append((type_name, type_code))

    monkeypatch.setattr(tvm_datatype, "register", register)
    return registered


def test_register_custom_dtype_without_exception_probe(monkeypatch):
    entries = {}
    registered = _patch_custom_dtype_registry(monkeypatch, entries)
    monkeypatch.setattr(
        tvm_datatype,
        "get_type_code",
        lambda _: pytest.fail("get_type_code must not be used to probe an unregistered dtype"),
    )

    dtypes._register_custom_dtype("test_mxfp", 131)

    assert registered == [("test_mxfp", 131)]


def test_register_custom_dtype_is_idempotent(monkeypatch):
    registered = _patch_custom_dtype_registry(monkeypatch, {131: "test_mxfp"})

    dtypes._register_custom_dtype("test_mxfp", 131)

    assert registered == []


def test_register_custom_dtype_rejects_name_conflict(monkeypatch):
    _patch_custom_dtype_registry(monkeypatch, {132: "test_mxfp"})

    with pytest.raises(RuntimeError, match="name is already registered with code 132"):
        dtypes._register_custom_dtype("test_mxfp", 131)


def test_register_custom_dtype_rejects_code_conflict(monkeypatch):
    _patch_custom_dtype_registry(monkeypatch, {131: "another_dtype"})

    with pytest.raises(RuntimeError, match="code is already registered for 'another_dtype'"):
        dtypes._register_custom_dtype("test_mxfp", 131)
