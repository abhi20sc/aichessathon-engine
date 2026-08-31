"""FEN <-> internal state, and a perft driver for correctness testing."""
import numpy as np
import numpy.typing as npt

from .core import gen_moves, make, perft
from .zobrist import CASTLE_KEY, EP_KEY, PIECE_KEY, SIDE_KEY

U = np.uint64
EMPTY = 12
PIECE_CHARS = "PNBRQKpnbrqk"
MAXPLY = 128


def new_stack(maxply: int = MAXPLY) -> tuple[
        npt.NDArray[np.uint64], npt.NDArray[np.int8], npt.NDArray[np.uint32]]:
    return (np.zeros((maxply, 19), dtype=U),
            np.full((maxply, 64), EMPTY, dtype=np.int8),
            np.zeros((maxply, 256), dtype=np.uint32))


def set_fen(s: npt.NDArray[np.uint64], mb: npt.NDArray[np.int8], fen: str) -> None:
    s[:] = 0
    mb[:] = EMPTY
    parts = fen.split()
    rows = parts[0].split("/")
    for r, row in enumerate(rows):
        rank = 7 - r
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                sq = rank * 8 + f
                pc = PIECE_CHARS.index(ch)
                s[pc] |= U(1) << U(sq)
                mb[sq] = pc
                f += 1
    for pc in range(6):
        s[12] |= s[pc]
        s[13] |= s[pc + 6]
    s[14] = 0 if parts[1] == "w" else 1
    cr = 0
    if len(parts) > 2:
        for ch, bitv in (("K", 1), ("Q", 2), ("k", 4), ("q", 8)):
            if ch in parts[2]:
                cr |= bitv
    s[15] = cr
    ep = parts[3] if len(parts) > 3 else "-"
    s[16] = 64 if ep == "-" else (ord(ep[0]) - 97) + (int(ep[1]) - 1) * 8
    s[17] = int(parts[4]) if len(parts) > 4 else 0
    s[18] = zobrist(s, mb)


def zobrist(s: npt.NDArray[np.uint64], mb: npt.NDArray[np.int8]) -> np.uint64:
    """Compute a position hash from scratch. Only used when setting a FEN;
    every other position gets its hash incrementally from make()."""
    h = U(0)
    for sq in range(64):
        pc = int(mb[sq])
        if pc != EMPTY:
            h ^= PIECE_KEY[pc, sq]
    h ^= CASTLE_KEY[int(s[15])]
    if int(s[16]) != 64:
        h ^= EP_KEY[int(s[16]) & 7]
    if int(s[14]) == 1:
        h ^= SIDE_KEY
    return np.uint64(h)


def run_perft(fen: str, depth: int) -> int:
    st, mbs, buf = new_stack()
    set_fen(st[0], mbs[0], fen)
    return perft(st, mbs, buf, 0, depth)


def perft_divide(fen: str, depth: int) -> dict[str, int]:
    """Per-root-move counts, for pinpointing exactly where a bug lives."""
    st, mbs, buf = new_stack()
    set_fen(st[0], mbs[0], fen)
    n = gen_moves(st[0], mbs[0], buf[0])
    out = {}
    for i in range(n):
        mv = buf[0][i]
        if make(st[0], mbs[0], st[1], mbs[1], mv):
            frm = int(mv) & 63
            to = (int(mv) >> 6) & 63
            promo = (int(mv) >> 12) & 7
            uci = (chr(97 + frm % 8) + str(frm // 8 + 1) +
                   chr(97 + to % 8) + str(to // 8 + 1) +
                   ("" if promo == 0 else "nbrq"[promo - 1]))
            out[uci] = perft(st, mbs, buf, 1, depth - 1) if depth > 1 else 1
    return out
