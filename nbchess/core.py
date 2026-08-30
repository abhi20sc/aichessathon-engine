"""Numba-JIT bitboard core: move generation, make/unmake, perft.

State layout - one row of `st` per ply, dtype uint64, width 18:
    0..11  piece bitboards: WP WN WB WR WQ WK BP BN BB BR BQ BK
    12     white occupancy
    13     black occupancy
    14     side to move (0 = white, 1 = black)
    15     castling rights, bit0 WK bit1 WQ bit2 BK bit3 BQ
    16     en-passant target square (64 = none)
    17     halfmove clock
    18     Zobrist hash of the position
Mailbox `mb` holds one int8 piece code per square (12 = empty), so we never
have to scan twelve bitboards to find out what sits on a square.

Move encoding, uint32:
    bits 0-5   from square
    bits 6-11  to square
    bits 12-14 promotion piece (0 none, 1 N, 2 B, 3 R, 4 Q)
    bits 15-17 flag (0 quiet, 1 en-passant, 2 castle, 3 double push)
"""
import numpy as np
from numba import njit
from numba.types import boolean, int8, int64, uint32, uint64

from .tables import (
    BISHOP_MAGIC_A,
    BISHOP_MASK,
    BISHOP_OFF,
    BISHOP_SHIFT,
    BISHOP_TABLE,
    CASTLE_MASK,
    DEBRUIJN,
    DEBRUIJN_IDX,
    KING_ATT,
    KNIGHT_ATT,
    PAWN_ATT,
    ROOK_MAGIC_A,
    ROOK_MASK,
    ROOK_OFF,
    ROOK_SHIFT,
    ROOK_TABLE,
)
from .zobrist import CASTLE_KEY, EP_KEY, PIECE_KEY, SIDE_KEY

U = np.uint64
EMPTY = 12
NONE_SQ = U(64)

_k = dict(cache=False, fastmath=False, nogil=True)


@njit(uint64(uint64), inline='always', **_k)
def lsb(b):
    """Index of the least significant set bit (de Bruijn multiplication)."""
    return uint64(DEBRUIJN_IDX[(((b & (uint64(0) - b)) * DEBRUIJN) >> uint64(58))])


@njit(uint64(uint64), inline='always', **_k)
def popcount(b):
    b = b - ((b >> uint64(1)) & uint64(0x5555555555555555))
    b = (b & uint64(0x3333333333333333)) + ((b >> uint64(2)) & uint64(0x3333333333333333))
    b = (b + (b >> uint64(4))) & uint64(0x0F0F0F0F0F0F0F0F)
    return (b * uint64(0x0101010101010101)) >> uint64(56)


@njit(uint64(uint64, uint64), inline='always', **_k)
def rook_att(sq, occ):
    i = ((occ & ROOK_MASK[sq]) * ROOK_MAGIC_A[sq]) >> ROOK_SHIFT[sq]
    return ROOK_TABLE[ROOK_OFF[sq] + i]


@njit(uint64(uint64, uint64), inline='always', **_k)
def bishop_att(sq, occ):
    i = ((occ & BISHOP_MASK[sq]) * BISHOP_MAGIC_A[sq]) >> BISHOP_SHIFT[sq]
    return BISHOP_TABLE[BISHOP_OFF[sq] + i]


@njit(uint64(uint64, uint64), inline='always', **_k)
def queen_att(sq, occ):
    return rook_att(sq, occ) | bishop_att(sq, occ)


@njit(boolean(uint64[:], uint64, uint64), **_k)
def attacked(s, sq, by):
    """Is `sq` attacked by side `by` in state row `s`?"""
    off = by * uint64(6)
    occ = s[12] | s[13]
    if PAWN_ATT[uint64(1) - by, sq] & s[off]:
        return True
    if KNIGHT_ATT[sq] & s[off + uint64(1)]:
        return True
    if KING_ATT[sq] & s[off + uint64(5)]:
        return True
    if bishop_att(sq, occ) & (s[off + uint64(2)] | s[off + uint64(4)]):
        return True
    return bool(rook_att(sq, occ) & (s[off + uint64(3)] | s[off + uint64(4)]))


@njit(uint32(uint64, uint64, uint64, uint64), inline='always', **_k)
def mk(frm, to, promo, flag):
    return uint32(frm | (to << uint64(6)) | (promo << uint64(12)) | (flag << uint64(15)))


@njit(int64(uint64[:], int8[:], uint32[:]), **_k)
def gen_moves(s, mb, out):
    """Generate pseudo-legal moves into `out`, return how many."""
    n = 0
    us = s[14]
    them = uint64(1) - us
    off = us * uint64(6)
    own = s[12 + us]
    opp = s[12 + them]
    occ = own | opp
    empty = ~occ

    # ---- pawns ----
    pawns = s[off]
    if us == uint64(0):
        one = (pawns << uint64(8)) & empty
        two = ((one & uint64(0x0000000000FF0000)) << uint64(8)) & empty
        uint64(8)
        rank8 = uint64(0xFF00000000000000)
    else:
        one = (pawns >> uint64(8)) & empty
        two = ((one & uint64(0x0000FF0000000000)) >> uint64(8)) & empty
        uint64(56)   # equivalent to -8 modulo 64 for the shift-back below
        rank8 = uint64(0x00000000000000FF)

    b = one
    while b:
        to = lsb(b); b &= b - uint64(1)
        frm = to - uint64(8) if us == uint64(0) else to + uint64(8)
        if (uint64(1) << to) & rank8:
            out[n] = mk(frm, to, uint64(4), uint64(0)); n += 1
            out[n] = mk(frm, to, uint64(3), uint64(0)); n += 1
            out[n] = mk(frm, to, uint64(2), uint64(0)); n += 1
            out[n] = mk(frm, to, uint64(1), uint64(0)); n += 1
        else:
            out[n] = mk(frm, to, uint64(0), uint64(0)); n += 1
    b = two
    while b:
        to = lsb(b); b &= b - uint64(1)
        frm = to - uint64(16) if us == uint64(0) else to + uint64(16)
        out[n] = mk(frm, to, uint64(0), uint64(3)); n += 1

    ep = s[16]
    cap_targets = opp
    b = pawns
    while b:
        frm = lsb(b); b &= b - uint64(1)
        a = PAWN_ATT[us, frm] & cap_targets
        while a:
            to = lsb(a); a &= a - uint64(1)
            if (uint64(1) << to) & rank8:
                out[n] = mk(frm, to, uint64(4), uint64(0)); n += 1
                out[n] = mk(frm, to, uint64(3), uint64(0)); n += 1
                out[n] = mk(frm, to, uint64(2), uint64(0)); n += 1
                out[n] = mk(frm, to, uint64(1), uint64(0)); n += 1
            else:
                out[n] = mk(frm, to, uint64(0), uint64(0)); n += 1
        if ep != NONE_SQ and (PAWN_ATT[us, frm] & (uint64(1) << ep)):
            out[n] = mk(frm, ep, uint64(0), uint64(1)); n += 1

    # ---- knights ----
    b = s[off + uint64(1)]
    while b:
        frm = lsb(b); b &= b - uint64(1)
        a = KNIGHT_ATT[frm] & ~own
        while a:
            to = lsb(a); a &= a - uint64(1)
            out[n] = mk(frm, to, uint64(0), uint64(0)); n += 1

    # ---- bishops / rooks / queens ----
    b = s[off + uint64(2)]
    while b:
        frm = lsb(b); b &= b - uint64(1)
        a = bishop_att(frm, occ) & ~own
        while a:
            to = lsb(a); a &= a - uint64(1)
            out[n] = mk(frm, to, uint64(0), uint64(0)); n += 1
    b = s[off + uint64(3)]
    while b:
        frm = lsb(b); b &= b - uint64(1)
        a = rook_att(frm, occ) & ~own
        while a:
            to = lsb(a); a &= a - uint64(1)
            out[n] = mk(frm, to, uint64(0), uint64(0)); n += 1
    b = s[off + uint64(4)]
    while b:
        frm = lsb(b); b &= b - uint64(1)
        a = queen_att(frm, occ) & ~own
        while a:
            to = lsb(a); a &= a - uint64(1)
            out[n] = mk(frm, to, uint64(0), uint64(0)); n += 1

    # ---- king ----
    b = s[off + uint64(5)]
    ksq = lsb(b)
    a = KING_ATT[ksq] & ~own
    while a:
        to = lsb(a); a &= a - uint64(1)
        out[n] = mk(ksq, to, uint64(0), uint64(0)); n += 1

    # ---- castling ----
    cr = s[15]
    if us == uint64(0):
        if (cr & uint64(1)) and not (occ & uint64(0x60)):
            if not attacked(s, uint64(4), them) and not attacked(s, uint64(5), them):
                out[n] = mk(uint64(4), uint64(6), uint64(0), uint64(2)); n += 1
        if (cr & uint64(2)) and not (occ & uint64(0x0E)):
            if not attacked(s, uint64(4), them) and not attacked(s, uint64(3), them):
                out[n] = mk(uint64(4), uint64(2), uint64(0), uint64(2)); n += 1
    else:
        if (cr & uint64(4)) and not (occ & uint64(0x6000000000000000)):
            if not attacked(s, uint64(60), them) and not attacked(s, uint64(61), them):
                out[n] = mk(uint64(60), uint64(62), uint64(0), uint64(2)); n += 1
        if (cr & uint64(8)) and not (occ & uint64(0x0E00000000000000)):
            if not attacked(s, uint64(60), them) and not attacked(s, uint64(59), them):
                out[n] = mk(uint64(60), uint64(58), uint64(0), uint64(2)); n += 1
    return n


@njit(boolean(uint64[:], int8[:], uint64[:], int8[:], uint32), **_k)
def make(s, mb, ns, nmb, mv):
    """Apply `mv` to state (s, mb), writing the result into (ns, nmb).

    Maintains the Zobrist hash incrementally in slot 18. Returns False if the
    move leaves our own king in check, i.e. the move was pseudo-legal only.
    """
    for i in range(19):
        ns[i] = s[i]
    for i in range(64):
        nmb[i] = mb[i]

    frm = uint64(mv & uint32(63))
    to = uint64((mv >> uint32(6)) & uint32(63))
    promo = uint64((mv >> uint32(12)) & uint32(7))
    flag = uint64((mv >> uint32(15)) & uint32(7))

    us = s[14]
    them = uint64(1) - us
    pc = uint64(mb[frm])
    fb = uint64(1) << frm
    tb = uint64(1) << to
    h = s[18] ^ SIDE_KEY

    # retire the old en-passant file from the hash before setting a new one
    if s[16] != NONE_SQ:
        h ^= EP_KEY[s[16] & uint64(7)]
    ns[16] = NONE_SQ
    ns[17] = s[17] + uint64(1)

    if flag == uint64(1):                       # en passant
        csq = to - uint64(8) if us == uint64(0) else to + uint64(8)
        cap = uint64(nmb[csq])
        ns[cap] &= ~(uint64(1) << csq)
        ns[12 + them] &= ~(uint64(1) << csq)
        nmb[csq] = int8(EMPTY)
        h ^= PIECE_KEY[cap, csq]
        ns[17] = uint64(0)
    else:
        cap = uint64(nmb[to])
        if cap != uint64(EMPTY):
            ns[cap] &= ~tb
            ns[12 + them] &= ~tb
            h ^= PIECE_KEY[cap, to]
            ns[17] = uint64(0)

    ns[pc] &= ~fb
    ns[12 + us] &= ~fb
    nmb[frm] = int8(EMPTY)
    h ^= PIECE_KEY[pc, frm]

    if promo != uint64(0):
        newpc = us * uint64(6) + promo
        ns[newpc] |= tb
        nmb[to] = int8(newpc)
        h ^= PIECE_KEY[newpc, to]
        ns[17] = uint64(0)
    else:
        ns[pc] |= tb
        nmb[to] = int8(pc)
        h ^= PIECE_KEY[pc, to]
    ns[12 + us] |= tb

    if pc == us * uint64(6):                    # a pawn moved
        ns[17] = uint64(0)
    if flag == uint64(3):                       # double push
        ns[16] = (frm + to) >> uint64(1)
        h ^= EP_KEY[ns[16] & uint64(7)]

    if flag == uint64(2):                       # castling: hop the rook too
        if to == uint64(6):
            rf, rt = uint64(7), uint64(5)
        elif to == uint64(2):
            rf, rt = uint64(0), uint64(3)
        elif to == uint64(62):
            rf, rt = uint64(63), uint64(61)
        else:
            rf, rt = uint64(56), uint64(59)
        rpc = uint64(nmb[rf])
        ns[rpc] &= ~(uint64(1) << rf)
        ns[rpc] |= (uint64(1) << rt)
        ns[12 + us] &= ~(uint64(1) << rf)
        ns[12 + us] |= (uint64(1) << rt)
        nmb[rf] = int8(EMPTY)
        nmb[rt] = int8(rpc)
        h ^= PIECE_KEY[rpc, rf] ^ PIECE_KEY[rpc, rt]

    ns[15] = s[15] & uint64(CASTLE_MASK[frm]) & uint64(CASTLE_MASK[to])
    h ^= CASTLE_KEY[s[15]] ^ CASTLE_KEY[ns[15]]
    ns[14] = them
    ns[18] = h

    ksq = lsb(ns[us * uint64(6) + uint64(5)])
    return not attacked(ns, ksq, them)


@njit(boolean(uint64[:], int8[:], uint64[:], int8[:]), **_k)
def make_null(s, mb, ns, nmb):
    """Pass the move to the opponent. Used by null-move pruning."""
    for i in range(19):
        ns[i] = s[i]
    for i in range(64):
        nmb[i] = mb[i]
    h = s[18] ^ SIDE_KEY
    if s[16] != NONE_SQ:
        h ^= EP_KEY[s[16] & uint64(7)]
    ns[16] = NONE_SQ
    ns[17] = s[17] + uint64(1)
    ns[14] = uint64(1) - s[14]
    ns[18] = h
    return True


@njit(int64(uint64[:, :], int8[:, :], uint32[:, :], int64, int64), **_k)
def perft(stack, mbs, buf, ply, depth):
    if depth == 0:
        return 1
    n = gen_moves(stack[ply], mbs[ply], buf[ply])
    total = 0
    for i in range(n):
        if make(stack[ply], mbs[ply], stack[ply + 1], mbs[ply + 1], buf[ply][i]):
            if depth == 1:
                total += 1
            else:
                total += perft(stack, mbs, buf, ply + 1, depth - 1)
    return total
