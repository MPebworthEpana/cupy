from __future__ import annotations

import os

import numpy

INT32_MAX = int(numpy.iinfo(numpy.int32).max)
INT32_MIN = int(numpy.iinfo(numpy.int32).min)


class IndexWidthOverflowError(OverflowError):
    """Raised when an index/size exceeds int32 but CUPY_INDEX_WIDTH=32."""


def get_index_width_mode() -> str:
    """Return ``auto``, ``32``, or ``64`` from ``CUPY_INDEX_WIDTH``."""
    raw = os.environ.get('CUPY_INDEX_WIDTH')
    if raw is None or raw == 'auto':
        return 'auto'
    mode = raw.strip().lower()
    if mode in ('', 'auto'):
        return 'auto'
    if mode in ('32', 'int32'):
        return '32'
    if mode in ('64', 'int64'):
        return '64'
    raise ValueError(
        f'CUPY_INDEX_WIDTH must be auto, 32, or 64; got {raw!r}')


def _need_int64(*sizes, arrays=(), check_contents=True) -> bool:
    """True if any size or array value cannot be represented as int32."""
    for size in sizes:
        if size is None:
            continue
        ival = int(size)
        if ival > INT32_MAX or ival < INT32_MIN:
            return True

    if arrays is None:
        return False
    if not isinstance(arrays, (tuple, list)):
        arrays = (arrays,)

    for arr in arrays:
        if arr is None:
            continue
        dtype = getattr(arr, 'dtype', None)
        if dtype is None:
            continue
        dtype = numpy.dtype(dtype)
        if numpy.can_cast(dtype, numpy.int32, casting='safe'):
            continue
        if not check_contents:
            return True
        if int(getattr(arr, 'size', 0)) == 0:
            continue
        if not numpy.issubdtype(dtype, numpy.integer):
            return True
        # max first: the common overflow is a large positive index.
        if int(arr.max()) > INT32_MAX:
            return True
        if int(arr.min()) < INT32_MIN:
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
    if _need_int64(*sizes, arrays=arrays, check_contents=check_contents):
        if mode == '32':
            raise IndexWidthOverflowError(
                'Index or size exceeds int32.max but CUPY_INDEX_WIDTH=32')
        return numpy.int64
    return numpy.int32
