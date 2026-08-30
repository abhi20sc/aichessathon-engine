"""Precomputed attack tables. Built with numpy at import time (no JIT needed),
so this costs milliseconds, not seconds, against the 60s init budget.
"""
import numpy as np

from .magics import BISHOP_BITS, BISHOP_MAGIC, ROOK_BITS, ROOK_MAGIC

U = np.uint64
MASK64 = (1 << 64) - 1

ROOK_DIRS   = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _slide(sq, dirs, occ, stop_before_edge=False):
    f, r = sq % 8, sq // 8
    att = 0
    for df, dr in dirs:
        nf, nr = f + df, r + dr
        while 0 <= nf < 8 and 0 <= nr < 8:
            if stop_before_edge and not (0 <= nf + df < 8 and 0 <= nr + dr < 8):
                break
            att |= 1 << (nr * 8 + nf)
            if occ & (1 << (nr * 8 + nf)):
                break
            nf += df; nr += dr
    return att


def _subsets(m):
    out, s = [], 0
    while True:
        out.append(s)
        s = (s - m) & m
        if s == 0:
            return out


def _leaper(sq, deltas):
    f, r = sq % 8, sq // 8
    att = 0
    for df, dr in deltas:
        nf, nr = f + df, r + dr
        if 0 <= nf < 8 and 0 <= nr < 8:
            att |= 1 << (nr * 8 + nf)
    return att


KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KING_DELTAS   = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))

KNIGHT_ATT = np.array([_leaper(s, KNIGHT_DELTAS) for s in range(64)], dtype=U)
KING_ATT   = np.array([_leaper(s, KING_DELTAS)   for s in range(64)], dtype=U)

# PAWN_ATT[colour][square] - squares that pawn attacks
PAWN_ATT = np.zeros((2, 64), dtype=U)
for s in range(64):
    PAWN_ATT[0, s] = _leaper(s, ((-1, 1), (1, 1)))    # white captures upward
    PAWN_ATT[1, s] = _leaper(s, ((-1, -1), (1, -1)))  # black captures downward

# --- magic sliding tables ---
ROOK_MASK   = np.array([_slide(s, ROOK_DIRS,   0, True) for s in range(64)], dtype=U)
BISHOP_MASK = np.array([_slide(s, BISHOP_DIRS, 0, True) for s in range(64)], dtype=U)
ROOK_MAGIC_A   = np.array(ROOK_MAGIC,   dtype=U)
BISHOP_MAGIC_A = np.array(BISHOP_MAGIC, dtype=U)
ROOK_SHIFT   = np.array([64 - b for b in ROOK_BITS],   dtype=U)
BISHOP_SHIFT = np.array([64 - b for b in BISHOP_BITS], dtype=U)

ROOK_OFF   = np.zeros(64, dtype=U)
BISHOP_OFF = np.zeros(64, dtype=U)
_o = 0
for s in range(64):
    ROOK_OFF[s] = _o; _o += 1 << ROOK_BITS[s]
ROOK_TABLE = np.zeros(_o, dtype=U)
_o = 0
for s in range(64):
    BISHOP_OFF[s] = _o; _o += 1 << BISHOP_BITS[s]
BISHOP_TABLE = np.zeros(_o, dtype=U)

for s in range(64):
    m = int(ROOK_MASK[s])
    for occ in _subsets(m):
        idx = ((occ * ROOK_MAGIC[s]) & MASK64) >> (64 - ROOK_BITS[s])
        ROOK_TABLE[int(ROOK_OFF[s]) + idx] = _slide(s, ROOK_DIRS, occ)
    m = int(BISHOP_MASK[s])
    for occ in _subsets(m):
        idx = ((occ * BISHOP_MAGIC[s]) & MASK64) >> (64 - BISHOP_BITS[s])
        BISHOP_TABLE[int(BISHOP_OFF[s]) + idx] = _slide(s, BISHOP_DIRS, occ)

# Castling rights are cleared whenever a piece leaves or lands on these squares.
# bit0 = white O-O, bit1 = white O-O-O, bit2 = black O-O, bit3 = black O-O-O
CASTLE_MASK = np.full(64, 0b1111, dtype=np.uint8)
CASTLE_MASK[0]  = 0b1101   # a1 rook -> white loses queenside
CASTLE_MASK[7]  = 0b1110   # h1 rook -> white loses kingside
CASTLE_MASK[4]  = 0b1100   # e1 king -> white loses both
CASTLE_MASK[56] = 0b0111   # a8
CASTLE_MASK[63] = 0b1011   # h8
CASTLE_MASK[60] = 0b0011   # e8

_DB = 0x03f79d71b4cb0a89
DEBRUIJN = U(_DB)
DEBRUIJN_IDX = np.zeros(64, dtype=np.uint8)
for i in range(64):
    DEBRUIJN_IDX[(((1 << i) * _DB) & MASK64) >> 58] = i
