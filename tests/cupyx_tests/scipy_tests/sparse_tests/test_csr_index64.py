from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
from cupy._core._index_width import INT32_MAX, IndexWidthOverflowError
from cupyx.scipy import sparse


def _wide_csr():
    # One stored value whose column index and shape need int64.
    data = cupy.array([1.0], dtype=numpy.float64)
    indices = cupy.array([INT32_MAX + 1], dtype=numpy.int64)
    indptr = cupy.array([0, 1], dtype=numpy.int64)
    return sparse.csr_matrix(
        (data, indices, indptr), shape=(1, INT32_MAX + 2))


class TestCsrIndexWidth:

    def test_small_stays_int32(self):
        data = cupy.array([1.0], dtype=numpy.float64)
        indices = cupy.array([0], dtype=numpy.int64)
        indptr = cupy.array([0, 1], dtype=numpy.int64)
        mat = sparse.csr_matrix((data, indices, indptr), shape=(1, 1))
        assert mat.indices.dtype == numpy.int32
        assert mat.indptr.dtype == numpy.int32
        assert int(mat.indices[0]) == 0

    def test_wide_shape_keeps_int64_values(self):
        mat = _wide_csr()
        assert mat.indices.dtype == numpy.int64
        assert mat.indptr.dtype == numpy.int64
        assert int(mat.indices[0]) == INT32_MAX + 1
        assert mat.shape == (1, INT32_MAX + 2)

    def test_empty_wide_shape_uses_int64(self):
        mat = sparse.csr_matrix((1, INT32_MAX + 2), dtype=numpy.float64)
        assert mat.indptr.dtype == numpy.int64
        assert mat.indices.dtype == numpy.int64
        assert mat.shape == (1, INT32_MAX + 2)

    def test_force64_small_matrix(self, monkeypatch):
        monkeypatch.setenv('CUPY_INDEX_WIDTH', '64')
        data = cupy.array([1.0], dtype=numpy.float64)
        indices = cupy.array([0], dtype=numpy.int32)
        indptr = cupy.array([0, 1], dtype=numpy.int32)
        mat = sparse.csr_matrix((data, indices, indptr), shape=(1, 1))
        assert mat.indices.dtype == numpy.int64
        assert mat.indptr.dtype == numpy.int64

    def test_force32_raises_on_wide_index(self, monkeypatch):
        monkeypatch.setenv('CUPY_INDEX_WIDTH', '32')
        data = cupy.array([1.0], dtype=numpy.float64)
        indices = cupy.array([INT32_MAX + 1], dtype=numpy.int64)
        indptr = cupy.array([0, 1], dtype=numpy.int64)
        with pytest.raises(IndexWidthOverflowError):
            sparse.csr_matrix(
                (data, indices, indptr), shape=(1, INT32_MAX + 2))

    def test_csr2dense_force64_small(self, monkeypatch):
        monkeypatch.setenv('CUPY_INDEX_WIDTH', '64')
        data = cupy.array([2.0], dtype=numpy.float64)
        indices = cupy.array([1], dtype=numpy.int32)
        indptr = cupy.array([0, 1], dtype=numpy.int32)
        mat = sparse.csr_matrix((data, indices, indptr), shape=(1, 3))
        dense = mat.todense()
        testing.assert_array_equal(dense, cupy.array([[0.0, 2.0, 0.0]]))

    def test_axis_scan_force64_matches_numpy(self, monkeypatch):
        monkeypatch.setenv('CUPY_INDEX_WIDTH', '64')
        a = cupy.arange(20, dtype=cupy.int64).reshape(4, 5)
        got = cupy.cumsum(a, axis=1)
        expect = numpy.cumsum(a.get(), axis=1)
        testing.assert_array_equal(got, expect)
