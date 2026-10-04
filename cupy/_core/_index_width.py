from __future__ import annotations

import os

import numpy

INT32_MAX = int(numpy.iinfo(numpy.int32).max)
INT32_MIN = int(numpy.iinfo(numpy.int32).min)

_VALID_MODES = frozenset({'auto', '32', '64'})


class IndexWidthOverflowError(OverflowError):
    """Raised when an index/size exceeds int32 but CUPY_INDEX_WIDTH=32."""


def get_index_width_mode() -> str:
    """Return ``auto``, ``32``, or ``64`` from ``CUPY_INDEX_WIDTH``."""
    raw = os.environ.get('CUPY_INDEX_WIDTH', 'auto')
    if raw is None:
        return 'auto'
    mode = str(raw).strip().lower()
    if mode in ('', 'auto', 'default'):
        return 'auto'
    if mode in ('32', 'int32', 'i32'):
        return '32'
    if mode in ('64', 'int64', 'i64'):
        return '64'
    raise ValueError(
        f'CUPY_INDEX_WIDTH must be auto, 32, or 64; got {raw!r}')


def _as_int(value):
    if value is None:
        return None
    if hasattr(value, 'item'):
        value = value.item()
    return int(value)


def sizes_need_int64(*sizes, arrays=(), check_contents=True) -> bool:
    """True if any size or array value cannot be represented as int32.

    This ignores ``CUPY_INDEX_WIDTH``. Use :func:`choose_index_dtype` to
    apply auto / force32 / force64.
    """
    for size in sizes:
        ival = _as_int(size)
        if ival is None:
            continue
        if ival > INT32_MAX or ival < INT32_MIN:
            return True

    if isinstance(arrays, numpy.ndarray):
        arrays = (arrays,)
    elif arrays is None:
        arrays = ()

    for arr in arrays:
        if arr is None:
            continue
        # Avoid importing cupy here (circular). ndarray-like is enough.
        dtype = getattr(arr, 'dtype', None)
        if dtype is None:
            continue
        dtype = numpy.dtype(dtype)
        if numpy.can_cast(dtype, numpy.int32, casting='safe'):
            continue
        if not check_contents:
            return True
        size = int(getattr(arr, 'size', 0))
        if size == 0:
            continue
        if not numpy.issubdtype(dtype, numpy.integer):
            return True
        maxval = _as_int(arr.max())
        minval = _as_int(arr.min())
        if minval < INT32_MIN or maxval > INT32_MAX:
            return True
    return False


def choose_index_dtype(*sizes, arrays=(), check_contents=True):
    """Pick ``numpy.int32`` or ``numpy.int64`` for scan/CSR index buffers.

    * ``CUPY_INDEX_WIDTH=auto`` (default): int32 if everything fits, else
      int64.
    * ``CUPY_INDEX_WIDTH=64``: always int64.
    * ``CUPY_INDEX_WIDTH=32``: int32, or :class:`IndexWidthOverflowError`
      if a size/value would overflow.
    """
    mode = get_index_width_mode()
    if mode == '64':
        return numpy.int64
    need64 = sizes_need_int64(
        *sizes, arrays=arrays, check_contents=check_contents)
    if need64:
        if mode == '32':
            raise IndexWidthOverflowError(
                'Index or size exceeds int32.max but CUPY_INDEX_WIDTH=32')
        return numpy.int64
    return numpy.int32


def index_kernel_ctype(dtype=None, *sizes, arrays=(), check_contents=True):
    """C type used in generated kernels: ``'int'`` or ``'long long'``."""
    if dtype is None:
        dtype = choose_index_dtype(
            *sizes, arrays=arrays, check_contents=check_contents)
    if numpy.dtype(dtype) == numpy.dtype(numpy.int64):
        return 'long long'
    return 'int'
