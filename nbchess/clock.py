"""A monotonic clock readable from inside compiled code.

Numba resolves ctypes function pointers in nopython mode, and a numpy array's
address is available there, so the search can poll a real deadline instead of
estimating one from a node budget. Measured cost is ~1 microsecond per poll;
at one poll per 2048 nodes that is under 0.1% of search time.
"""
import ctypes

import numpy as np
import numpy.typing as npt
from numba import njit
from numba.types import int64

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_clock_gettime = _libc.clock_gettime
_clock_gettime.argtypes = [ctypes.c_int, ctypes.c_void_p]
_clock_gettime.restype = ctypes.c_int

CLOCK_MONOTONIC = 1


def new_timebuf() -> npt.NDArray[np.int64]:
    """A two-slot struct timespec, allocated once and reused."""
    return np.zeros(2, dtype=np.int64)


@njit(int64(int64[::1]), cache=False, nogil=True)
def now_ns(buf: npt.NDArray[np.int64]) -> int:
    _clock_gettime(CLOCK_MONOTONIC, buf.ctypes.data)
    return buf[0] * 1_000_000_000 + buf[1]


def available() -> bool:
    """True if the compiled clock works. If it ever fails we fall back to a
    pure node budget rather than losing on time."""
    try:
        buf = new_timebuf()
        return now_ns(buf) > 0
    except Exception:
        return False
