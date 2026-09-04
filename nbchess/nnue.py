"""A small neural evaluation, trained by tools.nnue_train on positions this
team labelled with a reference engine (allowed: the ban covers what ships,
and what ships here is a 400 KB table of our own numbers).

Inputs are 768 piece-square features per perspective: own pieces first, then
the opponent's, the board flipped for Black. One shared hidden layer with a
clipped ReLU, then a linear output over the side-to-move accumulator followed
by the other side's. Output is centipawns from the side to move's view.

The accumulators are rebuilt from the board on every call. That is the slow
way round - an incremental update in make() is the classic trick - but it
keeps the network out of the move generator, and at this size it costs a
few microseconds.
"""
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numba import njit
from numba.core.types import float32, int8, int32, uint64

_PATH = Path(__file__).with_name("nnue.npz")
USE_NNUE = _PATH.exists()

if USE_NNUE:
    with np.load(_PATH) as _z:
        W1: npt.NDArray[np.float32] = np.ascontiguousarray(_z["w1"], dtype=np.float32)
        B1: npt.NDArray[np.float32] = np.ascontiguousarray(_z["b1"], dtype=np.float32)
        W2: npt.NDArray[np.float32] = np.ascontiguousarray(_z["w2"], dtype=np.float32)
        B2 = float(_z["b2"])
else:  # placeholders so the module still compiles; never used
    W1 = np.zeros((769, 8), dtype=np.float32)
    B1 = np.zeros(8, dtype=np.float32)
    W2 = np.zeros(16, dtype=np.float32)
    B2 = 0.0
HIDDEN = int(B1.shape[0])


@njit(int32(uint64[:], int8[:]), cache=False, nogil=True)
def nn_eval(s: npt.NDArray[np.uint64], mb: npt.NDArray[np.int8]) -> np.int32:
    acc_w = np.empty(HIDDEN, dtype=np.float32)
    acc_b = np.empty(HIDDEN, dtype=np.float32)
    for i in range(HIDDEN):
        acc_w[i] = B1[i]
        acc_b[i] = B1[i]
    for sq in range(64):
        pc = mb[sq]
        if pc == 12:
            continue
        colour = pc // 6
        ptype = pc % 6
        iw = colour * 384 + ptype * 64 + sq
        ib = (1 - colour) * 384 + ptype * 64 + (sq ^ 56)
        for i in range(HIDDEN):
            acc_w[i] += W1[iw, i]
            acc_b[i] += W1[ib, i]
    out = float32(B2)
    if s[14] == uint64(0):
        for i in range(HIDDEN):
            u = min(max(acc_w[i], float32(0.0)), float32(1.0))
            t = min(max(acc_b[i], float32(0.0)), float32(1.0))
            out += W2[i] * u + W2[HIDDEN + i] * t
    else:
        for i in range(HIDDEN):
            u = min(max(acc_b[i], float32(0.0)), float32(1.0))
            t = min(max(acc_w[i], float32(0.0)), float32(1.0))
            out += W2[i] * u + W2[HIDDEN + i] * t
    if out > float32(3000.0):
        out = float32(3000.0)
    if out < float32(-3000.0):
        out = float32(-3000.0)
    return int32(out)
