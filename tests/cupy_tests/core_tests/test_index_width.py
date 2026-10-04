from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy
import pytest


def _load_helper():
    # Load from source so this test does not require a rebuilt CuPy extension.
    helper = (
        Path(__file__).resolve().parents[3] / 'cupy' / '_core' / '_index_width.py'
    )
    spec = importlib.util.spec_from_file_location(
        'cupy_index_width_under_test', helper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def iw(monkeypatch):
    mod = _load_helper()

    def _set(mode):
        if mode is None:
            monkeypatch.delenv('CUPY_INDEX_WIDTH', raising=False)
        else:
            monkeypatch.setenv('CUPY_INDEX_WIDTH', mode)

    _set(None)
    return mod, _set


class TestChooseIndexDtype:

    def test_auto_small_is_int32(self, iw):
        mod, _set = iw
        _set('auto')
        assert mod.choose_index_dtype(0) == numpy.int32
        assert mod.choose_index_dtype(mod.INT32_MAX) == numpy.int32
        assert mod.choose_index_dtype(1, 2, 3) == numpy.int32

    def test_auto_promotes_at_int32_max_plus_one(self, iw):
        mod, _set = iw
        _set('auto')
        assert mod.choose_index_dtype(mod.INT32_MAX + 1) == numpy.int64
        assert mod.choose_index_dtype(-mod.INT32_MAX - 2) == numpy.int64

    def test_force64_always_int64(self, iw):
        mod, _set = iw
        _set('64')
        assert mod.choose_index_dtype(1) == numpy.int64
        assert mod.choose_index_dtype(mod.INT32_MAX + 1) == numpy.int64

    def test_force32_raises_on_overflow(self, iw):
        mod, _set = iw
        _set('32')
        assert mod.choose_index_dtype(10) == numpy.int32
        with pytest.raises(mod.IndexWidthOverflowError):
            mod.choose_index_dtype(mod.INT32_MAX + 1)

    def test_invalid_mode(self, iw, monkeypatch):
        mod, _set = iw
        monkeypatch.setenv('CUPY_INDEX_WIDTH', 'nope')
        with pytest.raises(ValueError, match='CUPY_INDEX_WIDTH'):
            mod.get_index_width_mode()

    def test_array_values_check_contents(self, iw):
        mod, _set = iw
        _set('auto')
        small = numpy.array([0, 1, 2], dtype=numpy.int64)
        assert mod.choose_index_dtype(
            arrays=(small,), check_contents=True) == numpy.int32
        assert mod.choose_index_dtype(
            arrays=(small,), check_contents=False) == numpy.int64
        wide = numpy.array([mod.INT32_MAX + 1], dtype=numpy.int64)
        assert mod.choose_index_dtype(
            arrays=(wide,), check_contents=True) == numpy.int64

    def test_index_kernel_ctype(self, iw):
        mod, _set = iw
        _set('auto')
        assert mod.index_kernel_ctype(None, 8) == 'int'
        assert mod.index_kernel_ctype(None, mod.INT32_MAX + 1) == 'long long'
        _set('64')
        assert mod.index_kernel_ctype(None, 8) == 'long long'
