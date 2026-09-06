"""A small neural evaluation, trained by tools.nnue_train on positions this
team labelled with a reference engine (allowed: the ban covers what ships,
and what ships here is a 400 KB table of our own numbers).

Inputs are 768 piece-square features per perspective: own pieces first, then
the opponent's, the board flipped for Black. One shared hidden layer with a
clipped ReLU, then a linear output over the side-to-move accumulator followed
by the other side's. Output is centipawns from the side to move's view.

The search keeps the accumulators up to date incrementally (acc_update after
every move, acc_copy across a null move) and reads the output from them; the
from-scratch path is kept for tools and tests. Weights ship as a
.safetensors file we wrote ourselves, read here with numpy alone.
"""
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numba import njit
from numba.core.types import float32, int8, int32, int64, uint32, uint64

_PATH = Path(__file__).with_name("nnue.safetensors")
USE_NNUE = _PATH.exists()


def load_safetensors(path: Path) -> dict[str, npt.NDArray[np.float32]]:
    """Read a .safetensors file with numpy alone: an 8-byte little-endian
    header length, a JSON header naming each tensor's dtype, shape and byte
    range, then the raw data. Only float32 tensors are written by our tools."""
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n].decode("utf-8"))
    base = 8 + n
    out: dict[str, npt.NDArray[np.float32]] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] != "F32":
            raise ValueError(f"{path}: tensor {name} is {meta['dtype']}, expected F32")
        lo, hi = meta["data_offsets"]
        arr = np.frombuffer(raw[base + lo:base + hi], dtype="<f4").reshape(meta["shape"])
        out[name] = np.ascontiguousarray(arr, dtype=np.float32)
    return out


def save_safetensors(path: Path, tensors: dict[str, npt.NDArray[np.float32]]) -> None:
    """Write float32 tensors in the safetensors layout (tools only)."""
    header: dict[str, object] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        b = a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape),
                        "data_offsets": [offset, offset + len(b)]}
        chunks.append(b)
        offset += len(b)
    h = json.dumps(header).encode("utf-8")
    h += b" " * ((8 - len(h) % 8) % 8)
    path.write_bytes(len(h).to_bytes(8, "little") + h + b"".join(chunks))


if USE_NNUE:
    _z = load_safetensors(_PATH)
    W1: npt.NDArray[np.float32] = _z["w1"]
    B1: npt.NDArray[np.float32] = _z["b1"]
    #: Output weights, one row per piece-count bucket (a net without buckets
    #: has a single row that serves every position).
    W2: npt.NDArray[np.float32] = np.ascontiguousarray(np.atleast_2d(_z["w2"]), dtype=np.float32)
    B2: npt.NDArray[np.float32] = np.ascontiguousarray(_z["b2"].reshape(-1), dtype=np.float32)
    #: A residual net corrects the hand evaluation instead of replacing it.
    RESIDUAL = bool(int(_z["residual"][0])) if "residual" in _z else False
    #: Multiplier on the network's output; below 1 damps a noisy net.
    OUT_SCALE = float(_z["scale"][0]) if "scale" in _z else 1.0
else:  # placeholders so the module still compiles; never used
    W1 = np.zeros((769, 8), dtype=np.float32)
    B1 = np.zeros(8, dtype=np.float32)
    W2 = np.zeros((1, 16), dtype=np.float32)
    B2 = np.zeros(1, dtype=np.float32)
    RESIDUAL = False
    OUT_SCALE = 1.0
HIDDEN = int(B1.shape[0])
N_BUCKETS = int(W2.shape[0])
#: Piece counts (both sides, kings included) at or below which each bucket
#: applies; the last bucket takes everything above the last edge.
BUCKET_EDGES = np.array([12, 22], dtype=np.int64)


@njit(int64(int8[:], float32[:, ::1]), cache=False, nogil=True)
def acc_refresh(mb: npt.NDArray[np.int8], acc: npt.NDArray[np.float32]) -> int:
    """Both perspectives' accumulators from scratch: acc[0] is White's view,
    acc[1] Black's. Done once at the root; the search updates incrementally."""
    for i in range(HIDDEN):
        acc[0, i] = B1[i]
        acc[1, i] = B1[i]
    for sq in range(64):
        pc = mb[sq]
        if pc == 12:
            continue
        colour = pc // 6
        ptype = pc % 6
        iw = colour * 384 + ptype * 64 + sq
        ib = (1 - colour) * 384 + ptype * 64 + (sq ^ 56)
        for i in range(HIDDEN):
            acc[0, i] += W1[iw, i]
            acc[1, i] += W1[ib, i]
    return 0


@njit(int64(float32[:, ::1], int64, int64, int64), forceinline=True, cache=False, nogil=True)
def _acc_piece(acc: npt.NDArray[np.float32], pc: int, sq: int, sign: int) -> int:
    """Add (sign +1) or remove (sign -1) piece `pc` on `sq` in both views."""
    colour = pc // 6
    ptype = pc % 6
    iw = colour * 384 + ptype * 64 + sq
    ib = (1 - colour) * 384 + ptype * 64 + (sq ^ 56)
    if sign > 0:
        for i in range(HIDDEN):
            acc[0, i] += W1[iw, i]
            acc[1, i] += W1[ib, i]
    else:
        for i in range(HIDDEN):
            acc[0, i] -= W1[iw, i]
            acc[1, i] -= W1[ib, i]
    return 0


@njit(int64(float32[:, ::1], float32[:, ::1], int8[:], uint64, uint32), cache=False, nogil=True)
def acc_update(parent: npt.NDArray[np.float32], child: npt.NDArray[np.float32],
               mb: npt.NDArray[np.int8], us: np.uint64, mv: np.uint32) -> int:
    """Child accumulators after `mv`, from the parent's and the PARENT's
    mailbox (the board before the move). Mirrors core.make piece by piece."""
    for i in range(HIDDEN):
        child[0, i] = parent[0, i]
        child[1, i] = parent[1, i]
    frm = int64(mv & uint32(63))
    to = int64((mv >> uint32(6)) & uint32(63))
    promo = int64((mv >> uint32(12)) & uint32(7))
    flag = int64((mv >> uint32(15)) & uint32(7))
    pc = int64(mb[frm])
    side = int64(us)
    if flag == 1:                                   # en passant
        csq = to - 8 if side == 0 else to + 8
        _acc_piece(child, int64(mb[csq]), csq, -1)
    else:
        cap = int64(mb[to])
        if cap != 12:
            _acc_piece(child, cap, to, -1)
    _acc_piece(child, pc, frm, -1)
    if promo != 0:
        _acc_piece(child, side * 6 + promo, to, 1)
    else:
        _acc_piece(child, pc, to, 1)
    if flag == 2:                                   # castling: the rook hops too
        if to == 6:
            rf, rt = 7, 5
        elif to == 2:
            rf, rt = 0, 3
        elif to == 62:
            rf, rt = 63, 61
        else:
            rf, rt = 56, 59
        rpc = int64(mb[rf])
        _acc_piece(child, rpc, rf, -1)
        _acc_piece(child, rpc, rt, 1)
    return 0


@njit(int64(float32[:, ::1], float32[:, ::1]), cache=False, nogil=True)
def acc_copy(parent: npt.NDArray[np.float32], child: npt.NDArray[np.float32]) -> int:
    """A null move changes no piece: the child inherits the accumulators."""
    for i in range(HIDDEN):
        child[0, i] = parent[0, i]
        child[1, i] = parent[1, i]
    return 0


@njit(int64(int64), forceinline=True, cache=False, nogil=True)
def bucket_of(pieces: int) -> int:
    """Which output row scores a position with this many pieces on the board."""
    if N_BUCKETS == 1:
        return 0
    for k in range(BUCKET_EDGES.shape[0]):
        if pieces <= BUCKET_EDGES[k]:
            return min(k, N_BUCKETS - 1)
    return N_BUCKETS - 1


@njit(int32(uint64, float32[:, ::1], int64), cache=False, nogil=True)
def nn_output(stm: np.uint64, acc: npt.NDArray[np.float32], bucket: int) -> np.int32:
    """Centipawns from the side to move's view, given ready accumulators and
    the output row for the position's piece count."""
    out = B2[bucket]
    if stm == uint64(0):
        for i in range(HIDDEN):
            u = min(max(acc[0, i], float32(0.0)), float32(1.0))
            t = min(max(acc[1, i], float32(0.0)), float32(1.0))
            out += W2[bucket, i] * u + W2[bucket, HIDDEN + i] * t
    else:
        for i in range(HIDDEN):
            u = min(max(acc[1, i], float32(0.0)), float32(1.0))
            t = min(max(acc[0, i], float32(0.0)), float32(1.0))
            out += W2[bucket, i] * u + W2[bucket, HIDDEN + i] * t
    out *= float32(OUT_SCALE)
    if out > float32(3000.0):
        out = float32(3000.0)
    if out < float32(-3000.0):
        out = float32(-3000.0)
    return int32(out)


@njit(int32(uint64[:], int8[:]), cache=False, nogil=True)
def nn_eval(s: npt.NDArray[np.uint64], mb: npt.NDArray[np.int8]) -> np.int32:
    """Slow path: rebuild the accumulators and evaluate. Used for testing and
    by anything that has no accumulator to hand."""
    acc = np.empty((2, HIDDEN), dtype=np.float32)
    acc_refresh(mb, acc)
    pieces = 0
    for sq in range(64):
        if mb[sq] != 12:
            pieces += 1
    return nn_output(s[14], acc, bucket_of(pieces))
